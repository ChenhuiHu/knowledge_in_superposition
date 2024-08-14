from copy import deepcopy
from typing import List

import torch

from util import nethook


def get_reprs_at_word_tokens(model, tok, context_templates, words, layer, module_template, subtoken, track = "in"):
    """
    Retrieves the last token representation of `word` in `context_template`.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    
    for idx, context_template, word in zip(idxs, context_templates, words):
        sentence = context_template.format(word)
        #print(f"======>Get k value in [[{sentence}]] at [[{tok.decode(tok(sentence)['input_ids'][idx[0]])}]]<======")
    
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )


def get_words_idxs_in_templates(tok, context_templates, words, subtoken):
    """
    Computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    if subtoken == "last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (2 if "Llama-3-8B" in tok.name_or_path or ("Llama-2-7b-hf" in tok.name_or_path and prefixes_len[i]==1) or ("Llama-2-13b-hf" in tok.name_or_path and prefixes_len[i]==1)
                   else (3 if "Llama-2-7b-hf" in tok.name_or_path or "Llama-2-13b-hf" in tok.name_or_path else 1))
            ]
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(model, tok, contexts, idxs, layer, module_template, track = "in"):
    """
    Runs input through model and returns averaged representations of the tokens at each index in `idxs`.
    
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))
        
    for batch_contexts, batch_idxs in _batch(n=512):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)
        
        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
