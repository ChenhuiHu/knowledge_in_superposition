import os
from pathlib import Path
from typing import Dict, List

import torch

#from repr_tools import *
import repr_tools
from layer_stats import *

# Cache variables
inv_mom2_cache = {}


def determine_tmp_name(model_name):
    """Determine the layer name based on the model name and layer number."""
    if "gpt2" in model_name:
        return "transformer.h.{}.mlp.c_proj"
    elif "gpt-j" in model_name:
        return "transformer.h.{}.mlp.fc_out"
    elif "pythia" in model_name:
        return "gpt_neox.layers.{}.mlp.dense_4h_to_h"
    elif "Llama" in model_name:
        return "model.layers.{}.mlp.down_proj"
    

def get_inv_cov(model, tok, layer_name):
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            stats_dir="data/stats",
            ds_name="wikipedia",
            to_collect=["mom2"],
            sample_size=100000,
            precision="float32",
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_C_k(model, tok, request, layer, context_templates, args):
    """
    Computes the C^{-1} @ k.
    """

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=determine_tmp_name(model.config._name_or_path),
        track="in",
    )
    if "subject_" in args.fact_token and args.fact_token.index("subject_") == 0:
        word = request["subject"]
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=args.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif args.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
    else:
        raise ValueError(f"fact_token={args.fact_token} not recognized")

    # Apply inverse second moment adjustment
    k = cur_repr
    C_k = get_inv_cov(
        model,
        tok,
        determine_tmp_name(model.config._name_or_path).format(layer),
    ) @ k.unsqueeze(1)
    C_k = C_k.squeeze()

    return C_k / C_k.norm()


def compute_k(model, tok, request, layer, context_templates, args):
    """
    Computes k.
    """

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=determine_tmp_name(model.config._name_or_path),
        track="in",
    )
    if "subject_" in args.fact_token and args.fact_token.index("subject_") == 0:
        word = request["subject"]
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=args.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif args.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
    else:
        raise ValueError(f"fact_token={args.fact_token} not recognized")
    
    return cur_repr


def compute_k_C_k(model, tok, layer, ki, kj):
    """
    Computes the ki @ C^{-1} @ kj.
    """
    C_kj = get_inv_cov(
        model,
        tok,
        determine_tmp_name(model.config._name_or_path).format(layer),
    ) @ kj.unsqueeze(1)
    C_kj = C_kj.squeeze()
    
    ki_C_kj = torch.dot(ki, C_kj)

    return ki_C_kj