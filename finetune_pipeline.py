from typing import Optional
from huggingface_hub import snapshot_download
from pathlib import Path
import os
import pandas as pd
import subprocess
from datasets import load_dataset


def download_data():
    mistral_models_path = "/local-scratch/nigam/users/aunell/model/mistral_models/7B-v0.3"
    os.makedirs(mistral_models_path, exist_ok=True)

    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3",token="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",\
         allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

def make_data():
    os.makedirs('/share/pi/nigam/users/aunell/mistral-finetune/data', exist_ok=True)
    # df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')
    df = load_dataset("HuggingFaceH4/ultrachat_200k")
    train_df = pd.DataFrame(df['train_sft'])

# Convert 'test' dataset into a pandas DataFrame
    # test_df = pd.DataFrame(df['test_sft'])
    df_train=train_df.sample(frac=0.95,random_state=200)
    df_eval=train_df.drop(df_train.index)
    df_train.to_json("/share/pi/nigam/users/aunell/mistral-finetune/data/ultrachat_chunk_train.jsonl", orient="records", lines=True)
    df_eval.to_json("/share/pi/nigam/users/aunell/mistral-finetune/data/ultrachat_chunk_eval.jsonl", orient="records", lines=True)

def reformat_data(input_file: str):
    """
    Calls the reformat_data function from the utils module.

    Args:
        input_file (str): Path to the input JSONL file.
    """
    try:
        # Run the subprocess command
        subprocess.run(
            ["python", "-m", "utils.reformat_data", input_file],
            check=True
        )
        print(f"Successfully reformatted {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while reformatting: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def validate_data(train_yaml: str):
    """
    Calls the validate_data function from the utils module with the given YAML configuration.

    Args:
        train_yaml (str): Path to the YAML configuration file.
    """
    try:
        # Run the subprocess command
        subprocess.run(
            ["python", "-m", "utils.validate_data", "--train_yaml", train_yaml],
            check=True
        )
        print(f"Validation successful for: {train_yaml}")
    except subprocess.CalledProcessError as e:
        print(f"Validation failed: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def make_config():
    config = """
    # data
    data:
    instruct_data: "/content/data/ultrachat_chunk_train.jsonl"  # Fill
    data: ""  # Optionally fill with pretraining data
    eval_instruct_data: "/content/data/ultrachat_chunk_eval.jsonl"  # Optionally fill

    # model
    model_id_or_path: "/content/mistral_models"  # Change to downloaded path
    lora:
    rank: 64

    # optim
    # tokens per training steps = batch_size x num_GPUs x seq_len
    # we recommend sequence length of 32768
    # If you run into memory error, you can try reduce the sequence length
    seq_len: 8192
    batch_size: 1
    num_microbatches: 8
    max_steps: 100
    optim:
    lr: 1.e-4
    weight_decay: 0.1
    pct_start: 0.05

    # other
    seed: 0
    log_freq: 1
    eval_freq: 100
    no_eval: False
    ckpt_freq: 100

    save_adapters: True  # save only trained LoRA adapters. Set to `False` to merge LoRA adapter into the base model and save full fine-tuned model

    run_dir: "/content/test_ultra"  # Fill
    """

    # save the same file locally into the example.yaml file
    import yaml
    with open('example.yaml', 'w') as file:
        yaml.dump(yaml.safe_load(config), file)

def run_training(config_file: str, nproc_per_node: int = 1):
    """
    Runs the training script using torchrun with the specified configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.
        nproc_per_node (int): Number of processes to run per node (default is 1).
    """
    try:
        # Build the command
        command = [
            "torchrun",
            f"--nproc-per-node={nproc_per_node}",
            "-m", "train",
            config_file
        ]
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Training started successfully with config: {config_file}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
    
import wandb

def initialize_wandb(project_name: str, run_name: Optional[str] = None):
    if not project_name:
        raise ValueError("`wandb.project` must not be an empty string.")
    wandb.init(project=project_name, name=run_name)  

if __name__ == "__main__":
    initialize_wandb("mistral-finetune")
    # Sequence of steps to run the pipeline
    # print("Starting data download...")
    # download_data()
    # print("Data download completed. Preparing data...")
    # make_data()
    # breakpoint()
    # print("Data preparation completed. Reformatting data...")
    # reformat_data("/share/pi/nigam/users/aunell/mistral-finetune/data/ultrachat_chunk_train.jsonl")
    # reformat_data("/share/pi/nigam/users/aunell/mistral-finetune/data/ultrachat_chunk_eval.jsonl")
    # print("Reformatting completed. Validating data...")
    validate_data("/share/pi/nigam/users/aunell/mistral-finetune/example/7B.yaml")
    print("Validation completed. Creating configuration file...")
    breakpoint()
    make_config()
    print("Configuration file created. Starting training...")
    run_training("example.yaml")
    print("Training pipeline completed.")