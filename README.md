# Knowledge in Superposition

This is the relevant code for the paper, Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models.



# Requirements

```shell
conda create -n knowledge_in_superposition python=3.10
```

```shell
pip install -r requirements.txt
```



# Quick Start

Here, we take GPT2-Small as an example, allowing us to complete the experiments relatively quickly. (Using the largest Llama2-13B model might require running for about a week on an 80G A100 to obtain complete results, whereas GPT2-Small can achieve full results in 2-3 hours on a 24G 3090.) If you wish to use other models, simply modify the `--model_name` in the following command accordingly.

First, you need to cache the estimated covariance matrix, which serves as the starting point for everything:

```sh
python3 Cache_Cov.py --model_name=gpt2-small
```

By default, the covariance matrix of all layers' MLP activations for the GPT2-Small model is estimated on the Wikipedia `20220301.en` dataset. The calculation results are saved by default in the `data/stats` directory. Other major parameters available include `--start_layer` and `--end_layer`, which control the range of layers for caching the covariance matrix (default is from layer 0 to the last layer, e.g., 0 to 12 for GPT2-Small); `--batch_token` specifies the number of tokens per batch for computing the covariance matrix, defaulting to the max of model input length. For the Llama2 and Llama3 series, it is recommended to use `--batch_token=2048` (since these models have longer default input lengths).

Subsequently, a P matrix corresponding to 128*128 knowledge pairs will be calculated, as defined in the original paper:

```sh
python3 Compute_PMatrix.py --model_name=gpt2-small
```

Other major parameters available include `--start_layer` and `--end_layer`, which control the range of layers for calculating the P matrix (default is from layer 0 to the last layer, e.g., 0 to 12 for GPT2-Small), and similarly for other matrix computing.

Calculate the Q matrix corresponding to 128*128 knowledge pairs:

```sh
python3 Compute_QMatrix.py --model_name=gpt2-small
```

Calculate the matrix of angles between knowledge representations in the whitening space for 128*128 knowledge pairs:

```sh
python3 Compute_Whitening_Cos.py --model_name=gpt2-small
```

Calculate the matrix of angles between knowledge representations in the activation space for 128*128 knowledge pairs:

```sh
python3 Compute_Activation_Cos.py --model_name=gpt2-small
```



You can then conveniently use the example notebooks in the `notebooks` directory to observe the phenomenon of knowledge superposition.

