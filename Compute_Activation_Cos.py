import os
import torch
import argparse
import json
import typing
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from tqdm import tqdm
from compute import compute_k


class CounterFactDataset(Dataset):
    def __init__(self, data_dir: str, size: typing.Optional[int] = None):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]
        print(f"Loaded dataset from {data_dir} with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def check_gpu_availability():
    """Check for available GPUs and print their names."""
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        print(f"Found {gpu_count} GPU(s) on this machine.")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found on this machine.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Model Analysis Script")

    parser.add_argument(
        "--model_name",
        choices=[
            "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
            "EleutherAI/gpt-j-6B", "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b", "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf"
        ],
        default="gpt2-small",
        help="Model to investigate.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["counterfact"],
        default="counterfact",
        help="Dataset to perform.",
    )
    parser.add_argument(
        "--knowledge_type",
        choices=["known", "unknown", "mix"],
        default="known",
        help="Type of knowledge to perform.",
    )
    parser.add_argument(
        "--record_size",
        type=int,
        default=128,
        help="Size of dataset to record.",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="Starting layer for analysis.",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=None,
        help="Ending layer for analysis.",
    )
    parser.add_argument(
        "--fact_token",
        choices=["subject_last", "subject_first", "last"],
        default="subject_last",
        help="Position to record on.",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer based on the model name."""
    print("Instantiating model")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_dataset(knowledge_type):
    """Load dataset based on knowledge type."""
    file_map = {
        "known": "data/intersected_counterfact_known.json",
        "unknown": "data/intersected_counterfact_unknown.json",
    }
    if knowledge_type not in file_map:
        raise AttributeError("Mix type not available now...")
    print(f"Loading data from: {file_map[knowledge_type]}")
    return CounterFactDataset(file_map[knowledge_type])


def get_model_layers(model):
    """Get the number of layers in the model."""
    if hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        return model.config.n_layer
    else:
        raise AttributeError("The model config does not have 'num_hidden_layers' or 'n_layer' attribute.")


def cosine_similarity(k_i, k_j):
    """Compute cosine similarity between two vectors."""
    cosine_similarity = torch.dot(k_i, k_j) / (torch.norm(k_i) * torch.norm(k_j))
    return cosine_similarity.item()


def compute_cos_matrix(args, model, tokenizer, dataset):
    """Compute the Cosine Similarity Matrix for each layer."""
    context_templates = ['{}']
    print(f"Computing the Cosine Matrix for Activation from layer {args.start_layer} to layer {args.end_layer}.")

    for layer in range(args.start_layer, args.end_layer):
        print(f"Current Layer ====> {layer}")

        # Initialize the matrix to save the coefficients
        cos_matrix = np.empty((args.record_size, args.record_size))

        for i in tqdm(range(args.record_size), desc="Outer loop"):
            ki = compute_k(
                model,
                tokenizer,
                dataset[i]["requested_rewrite"],
                layer,
                context_templates,
                args
            )
            for j in tqdm(range(args.record_size), desc="Inner loop", leave=False):
                kj = compute_k(
                    model,
                    tokenizer,
                    dataset[j]["requested_rewrite"],
                    layer,
                    context_templates,
                    args
                )
                cos_matrix[i, j] = cosine_similarity(ki, kj)

        save_cos_matrix(args, cos_matrix, layer)


def save_cos_matrix(args, cos_matrix, layer):
    """Save the Cosine Similarity Matrix."""
    filename = f"{args.model_name}/{args.knowledge_type}_sample_size_{args.record_size}/layer_{layer}_subject_last_cos_matrix.npy"
    file_path = Path("data/activation_cos_matrix") / filename

    os.makedirs(file_path.parent, exist_ok=True)
    np.save(file_path, cos_matrix)
    print(f"Cosine Matrix saved to {file_path}")


def main():
    check_gpu_availability()
    args = parse_arguments()
    print(args)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    dataset = load_dataset(args.knowledge_type)

    if args.end_layer is None:
        args.end_layer = get_model_layers(model)

    compute_cos_matrix(args, model, tokenizer, dataset)


if __name__ == "__main__":
    main()
