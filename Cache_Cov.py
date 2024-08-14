import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.nethook import set_requires_grad

from layer_stats import layer_stats


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
    parser = argparse.ArgumentParser(description="Cov Collector")
    
    parser.add_argument(
        "--model_name", 
        default="gpt2-small", 
        choices=[
            "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl", 
            "EleutherAI/gpt-j-6B", "EleutherAI/pythia-1b", 
            "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", 
            "EleutherAI/pythia-6.9b", "meta-llama/Meta-Llama-3-8B", 
            "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-2-7b-hf", 
            "meta-llama/Llama-2-13b-hf"
        ],
    )
    parser.add_argument("--dataset", default="wikipedia", choices=["wikipedia"])
    parser.add_argument("--start_layer", default=0, type=int)
    parser.add_argument("--end_layer", default=None, type=int)
    parser.add_argument("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    parser.add_argument("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    parser.add_argument("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    parser.add_argument("--precision", default="float32", choices=["float64", "float32", "float16"])
    parser.add_argument("--stats_dir", default="data/stats")
    parser.add_argument("--download", default=0, type=int, choices=[0, 1])
    
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto').eval()
    return model, tokenizer


def get_model_layers(model):
    """Get the number of layers in the model."""
    if hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    elif hasattr(model.config, 'n_layer'):
        return model.config.n_layer
    else:
        raise AttributeError("The model config does not have 'num_hidden_layers' or 'n_layer' attribute.")

        
def collect_layer_stats(args, model, tokenizer):
    """Collect statistics for each layer in the model."""
    print(f"Computing the covariance from layer {args.start_layer} to layer {args.end_layer}.")
    
    for layer_num in range(args.start_layer, args.end_layer):
        
        layer_name = determine_layer_name(args.model_name, layer_num)
        
        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )

        
def determine_layer_name(model_name, layer_num):
    """Determine the layer name based on the model name and layer number."""
    if "gpt2" in model_name:
        return f"transformer.h.{layer_num}.mlp.c_proj"
    elif "gpt-j" in model_name:
        return f"transformer.h.{layer_num}.mlp.fc_out"
    elif "pythia" in model_name:
        return f"gpt_neox.layers.{layer_num}.mlp.dense_4h_to_h"
    elif "Llama" in model_name:
        return f"model.layers.{layer_num}.mlp.down_proj"

    
def main():
    check_gpu_availability()
    args = parse_arguments()
    print(args)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Disable gradient computation
    set_requires_grad(False, model)
    
    if args.end_layer is None:
        args.end_layer = get_model_layers(model)
    collect_layer_stats(args, model, tokenizer)

    
if __name__ == "__main__":
    main()
