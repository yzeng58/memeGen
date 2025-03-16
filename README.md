<h1 align="center"> <p>Humor-Aware AI: Evaluating and Improving LLMs in Meme Analysis</p></h1>
<h4 align="center">
    <p>
      <a href="https://yzeng58.github.io/" target="_blank">Yuchen Zeng</a><sup>*1</sup>, 
      <a href="https://scholar.google.com/citations?user=cvaTlWgAAAAJ&hl=en" target="_blank">Hemang Rajvanshy</a><sup>*1</sup>, 
      <a href="https://wonjunn.github.io/">Wonjun Kang</a><sup>*2</sup>, 
      <a href="https://jifanz.github.io/" target="_blank">Jifan Zhang</a><sup>1</sup>, 
      <a href="https://en.wikipedia.org/wiki/Bob_Mankoff" target="_blank">Bob Mankoff</a><sup>1</sup>, 
      <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>1</sup>
      <a href="https://easonnie.github.io/" target="_blank">Yixin Nie</a><sup>2</sup>, 
      <a href="https://yipin.github.io/" target="_blank">Yipin Zhou</a><sup>2</sup>, 
  </p>
  <p>
    <sup>1</sup>UW-Madison, <sup>2</sup> Meta
   </p>
    </h4>
<p align="center">
    <a href="https://github.com/UW-Madison-Lee-Lab/CoBSAT/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/UW-Madison-Lee-Lab/CoBSAT.svg">
    </a>
    <a href="https://arxiv.org/abs/2402.01293">
        <img alt="GitHub release" src="https://img.shields.io/badge/arXiv-2402.01293-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/datasets/yzeng58/CoBSAT">
        <img alt="Hugging Face" src="https://img.shields.io/badge/dataset-CoBSAT-orange">
    </a>
</p>

**Abstract**: The evolution from Large Language Models (LLMs) to Multimodal Large Language Models (MLLMs) has spurred research into extending In-Context Learning (ICL) to its multimodal counterpart. Existing such studies have primarily concentrated on image-to-text ICL. However, the Text-to-Image ICL (T2I-ICL), with its unique characteristics and potential applications, remains underexplored. To address this gap, we formally define the task of T2I-ICL and present CoBSAT, the first T2I-ICL benchmark dataset, encompassing ten tasks. Utilizing our dataset to benchmark six state-of-the-art MLLMs, we uncover considerable difficulties MLLMs encounter in solving T2I-ICL. We identify the primary challenges as the inherent complexity of multimodality and image generation. To overcome these challenges, we explore strategies like fine-tuning and Chain-of-Thought prompting, demonstrating notable improvements. Our code and dataset are available at <a href="https://github.com/UW-Madison-Lee-Lab/CoBSAT">this link</a>.

<img width="903" alt="image" src="imgs/t2i_icl.jpg">



# Benchmarking Meme Humor with Large Language Models

## Baseline Performance

### Single Meme Evaluation

```bash
# ours_v4
python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb

# Relca
python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb
```

### Pairwise Meme Comparison

```bash
# ours_v4 
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb

# relca
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb

```

<details>
<summary>Click to expand full experiments</summary>

### Pairwise Meme Comparison

#### Qwen-VL
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct
```

#### GPT
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o

python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o
```

#### Pixtral
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name pixtral-12b 

python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name pixtral-12b 
```

#### Gemini
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro

python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro
```

### Single Meme Evaluation 
#### Qwen-VL
```bash
python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-2B-Instruct

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-7B-Instruct

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-2B-Instruct

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-7B-Instruct

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct
```

#### GPT
```bash
python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o-mini

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o-mini

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o
```

#### Pixtral
```bash
python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b
```

#### Gemini
```bash
python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-flash

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-flash

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro 
```

#### LLama-3.2
```bash
python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-11B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-11B-Vision-Instruct

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-11B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-11B-Vision-Instruct

python evaluation.py --dataset_name ours_v4 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v4 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct

python evaluation.py --dataset_name relca_v2 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca_v2 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct
```
</details>

## Improving Meme Humor Benchmarking Without Model Updates

### Single Meme Evaluation

#### Chain of Thought

```bash
##############
# Multimodal #
##############

# use saved description
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb

#################
# Language-only #
#################

# use saved description
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb

```

<details>
<summary>Click to expand full experiments</summary>

##### Qwen2-VL-72B-Instruct
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
```

##### Llama-3.2-90B-Vision-Instruct
```bash
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2

python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4
```

##### Pixtral-12b
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name relca

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name relca_v2

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v4
```

##### GPT-4o
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v4

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v4
```

##### gemini-1.5-pro
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v4

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v4
```

##### Mixtral/Mistral
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-Large-Instruct-2407 --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-Large-Instruct-2407 --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-Large-Instruct-2407 --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-Large-Instruct-2407 --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-7B-Instruct-v0.3 --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-7B-Instruct-v0.3 --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-7B-Instruct-v0.3 --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-7B-Instruct-v0.3 --dataset_name ours_v4
```

##### Llama-3.1 
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-8B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-8B-Instruct --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-8B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-8B-Instruct --dataset_name ours_v4
```

##### Qwen2.5
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-14B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-14B-Instruct --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-14B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-14B-Instruct --dataset_name ours_v4

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4
```
</details>

#### In-Context Learning

```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb
```

<details>
<summary>Click to expand full experiments</summary>

##### Qwen2-VL-72B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v4
```

##### Llama-3.2-90B-Vision-Instruct
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v4
```

##### Qwen2.5-72B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
```

##### Mixtral-8x22B-Instruct-v0.1
```bash --- continue
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
```

##### Pixtral-12b
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name pixtral-12b --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name pixtral-12b --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name pixtral-12b --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name pixtral-12b --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name pixtral-12b --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name pixtral-12b --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name pixtral-12b --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name pixtral-12b --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 8 --model_name pixtral-12b --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 8  --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v4
```

##### GPT-4o
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gpt-4o --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gpt-4o --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gpt-4o --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gpt-4o --dataset_name relca_v2

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gpt-4o --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v4

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gpt-4o --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v4
```

##### gemini-1.5-pro
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gemini-1.5-pro --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gemini-1.5-pro --dataset_name relca_v2
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca_v2

python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gemini-1.5-pro --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v4

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gemini-1.5-pro --dataset_name ours_v4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v4
```
</details>

#### Incorporating Humor Theory for Rating Memes

```bash
########################################
# advanced model (able to output json) #
########################################

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb

################
# normal model #
################
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v5 --train_ml_model xgboost --data_mode both --n_pairs 2000 --wandb
```

<details>
<summary>Click to expand full experiments</summary>

##### Qwen2-VL-72B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct
```

##### Mixtral-8x22B-Instruct-v0.1
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct
```

##### Qwen2.5-72B-Instruct
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct
```

##### GPT-4o
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name gpt-4o

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name gpt-4o
```

##### Gemini-1.5-Pro
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca_v2 --model_name gemini-1.5-pro

python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v4 --model_name gemini-1.5-pro
```
</details>

### Pairwise Meme Comparison

#### Chain of Thought

```bash
##############
# Multimodal #
##############

# use saved description
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb

# produce cot 
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb

#################
# Language-only #
#################

# use saved description
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb

# produce cot 
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb
```

<details>
<summary>Click to expand full experiments</summary>

##### Qwen2-VL-72B-Instruct

```bash
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct--data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct
```


##### Gemini-1.5-Pro
```bash
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name gemini-1.5-pro

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name gemini-1.5-pro
```

##### GPT-4o
```bash
# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name gpt-4o
```

##### Pixtral-12B
```bash
# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name pixtral-12b

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name pixtral-12b

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name pixtral-12b

# TODO 
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name pixtral-12b

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name pixtral-12b

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name pixtral-12b

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name pixtral-12b

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name pixtral-12b
```

##### Mixtral-8x22B-Instruct-v0.1
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Mixtral-8x22B-Instruct-v0.1

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name Mixtral-8x22B-Instruct-v0.1

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Mixtral-8x22B-Instruct-v0.1 

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Mixtral-8x22B-Instruct-v0.1
```

##### Mistral-Large-Instruct-2407
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Mistral-Large-Instruct-2407

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name Mistral-Large-Instruct-2407

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Mistral-Large-Instruct-2407

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Mistral-Large-Instruct-2407
```

##### Mistral-7B-Instruct-v0.3
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Mistral-7B-Instruct-v0.3

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Mistral-7B-Instruct-v0.3 
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Llama-3.1-70B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name Llama-3.1-70B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Llama-3.1-70B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct  --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Llama-3.1-70B-Instruct
```

##### Llama-3.1-8B-Instruct
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Llama-3.1-8B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name Llama-3.1-8B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Llama-3.1-8B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct  --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Llama-3.1-8B-Instruct
```

##### Qwen2.5-14B-Instruct
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_2 --model_name Qwen2.5-14B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name Qwen2.5-14B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Qwen2.5-14B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Qwen2.5-14B-Instruct
```

##### Qwen2.5-72B-Instruct
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca_v2 --model_name Qwen2.5-72B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca_v2 --model_name Qwen2.5-72B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v4 --model_name Qwen2.5-72B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v4 --model_name Qwen2.5-72B-Instruct
```
</details>

#### In-Context Learning

```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb
```

<details>
<summary>Click to expand full experiments</summary>

##### Qwen2-VL-72B-Instruct
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name Qwen2-VL-72B-Instruct
```

##### Pixtral-12B
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name pixtral-12b 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name pixtral-12b 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name pixtral-12b 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name pixtral-12b 
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct
```

##### Qwen2.5-72B-Instruct
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct
```

##### Mixtral-8x22B-Instruct-v0.1
```bash -- continue
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct
```

##### GPT-4o
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca_v2 --model_name gpt-4o 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v4 --model_name gpt-4o 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name gpt-4o 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name gpt-4o 
```

##### Gemini-1.5-Pro
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca_v2 --model_name gemini-1.5-pro 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v4 --model_name gemini-1.5-pro 
```
</details>

#### Ensemble

```bash
# Example Ensemble -- need to correspondingly adjust description and context parameters 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --model_name Llama-3.1-70B-Instruct Qwen2-VL-72B-Instruct Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct '' Llama-3.2-90B-Vision-Instruct --context '' '' '' --wandb --not_load_model
```

<details>
<summary>Click to expand full experiments (TO BE ADDED)</summary>
</details>

## Enhancing Meme Humor Benchmarking Through Model Fine-Tuning

### Single Meme Evaluation

#### Standard

```bash
# one dataset
python finetune.py --eval_mode single --model_name Qwen2-VL-2B-Instruct
python evaluation.py --eval_mode single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test
python evaluation.py --model_name Qwen2-VL-2B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --not_load_model --wandb

# mixture
python finetune.py --eval_mode single --dataset_name relca ours_v4 
python evaluation.py --eval_mode single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 


python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb

```

<details>
<summary>Click to expand full experiments</summary>

###### Qwen2-VL-72B-Instruct
```bash
python finetune.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb

python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb


python finetune.py --eval_mode single --dataset_name relca ours_v4  --model_name Qwen2-VL-72B-Instruct
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 
python evaluation.py --prompt_name single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb
```

##### Pixtral-12B
```bash
python finetune.py --eval_mode single --model_name pixtral-12b --dataset_name relca
python evaluation.py --eval_mode single --model_name pixtral-12b --peft_variant qlora_relca_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --model_name pixtral-12b --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --eval_mode single --model_name pixtral-12b --peft_variant qlora_relca_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4
python evaluation.py --model_name pixtral-12b --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb

python finetune.py --eval_mode single --dataset_name relca ours_v4 --model_name pixtral-12b
python evaluation.py --eval_mode single --model_name pixtral-12b --peft_variant qlora_relca_mix_ours_v4_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 
python evaluation.py --eval_mode single --model_name pixtral-12b --peft_variant qlora_relca_mix_ours_v4_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --prompt_name single --model_name pixtral-12b --peft_variant qlora_relca_mix_ours_v4_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name pixtral-12b --peft_variant qlora_relca_mix_ours_v4_pixtral-12b_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb
```
</details>

##### Llama-3.1-70B-Instruct
```bash
python finetune.py --eval_mode single --model_name Llama-3.1-70B-Instruct --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Llama-3.1-70B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Llama-3.1-70B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct

python finetune.py --eval_mode single --dataset_name relca ours_v4  --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_mix_ours_v4_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_mix_ours_v4_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --prompt_name single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_mix_ours_v4_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --prompt_name single --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_mix_ours_v4_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
```

##### Qwen2.5-72B-Instruct
```bash
python finetune.py --eval_mode single --model_name Qwen2.5-72B-Instruct --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Qwen2.5-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Qwen2.5-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct

python finetune.py --eval_mode single --dataset_name relca ours_v4  --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --prompt_name single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --prompt_name single --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb --description Llama-3.2-90B-Vision-Instruct
```

### Pairwise Meme Comparison 

#### Standard 

```bash
# one dataset
python finetune.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct
python evaluation.py --model_name Qwen2-VL-2B-Instruct --eval_mode pairwise --peft_variant qlora_relca_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb

# mixture
python finetune.py --eval_mode pairwise --dataset_name relca ours_v4 
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --wandb


python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v4_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --not_load_model --wandb
```

<details>
<summary>Click to expand full experiments</summary>

###### Qwen2-VL-72B-Instruct
```bash
python finetune.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --dataset_name relca_v2
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --peft_variant qlora_relca_v2_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca_v2
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_v2_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb

python finetune.py --eval_mode pairwise --dataset_name ours_v4 relca_v2 --model_name Qwen2-VL-72B-Instruct
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca_v2 --wandb
```

##### Pixtral-12B
```bash
python finetune.py --eval_mode pairwise --model_name pixtral-12b --dataset_name relca_v2
python evaluation.py --model_name pixtral-12b --eval_mode pairwise --peft_variant qlora_relca_v2_pixtral-12b_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca_v2
python evaluation.py --eval_mode pairwise --model_name pixtral-12b --peft_variant qlora_relca_v2_pixtral-12b_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb

python finetune.py --eval_mode pairwise --dataset_name ours_v4 relca_v2 --model_name pixtral-12b
python evaluation.py --eval_mode pairwise --model_name pixtral-12b --peft_variant qlora_ours_v4_mix_relca_v2_pixtral-12b_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb
python evaluation.py --eval_mode pairwise --model_name pixtral-12b --peft_variant qlora_ours_v4_mix_relca_v2_pixtral-12b_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca_v2 --wandb
```

##### Llama-3.1-70B-Instruct
```bash
python finetune.py --eval_mode pairwise --model_name Llama-3.1-70B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Llama-3.1-70B-Instruct --eval_mode pairwise --peft_variant qlora_relca_v2_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Llama-3.1-70B-Instruct --peft_variant qlora_relca_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb --description Llama-3.2-90B-Vision-Instruct

python finetune.py --eval_mode pairwise --dataset_name ours_v4 relca_v2 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Llama-3.1-70B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Llama-3.1-70B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Llama-3.1-70B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca_v2 --wandb --description Llama-3.2-90B-Vision-Instruct
```

##### Qwen2.5-72B-Instruct
```bash
python finetune.py --eval_mode pairwise --model_name Qwen2.5-72B-Instruct --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --model_name Qwen2.5-72B-Instruct --eval_mode pairwise --peft_variant qlora_relca_v2_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca_v2 --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Qwen2.5-72B-Instruct --peft_variant qlora_relca_v2_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb --description Llama-3.2-90B-Vision-Instruct

python finetune.py --eval_mode pairwise --dataset_name ours_v4 relca_v2 --model_name Qwen2.5-72B-Instruct --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Qwen2.5-72B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v4 --wandb --description Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode pairwise --model_name Qwen2.5-72B-Instruct --peft_variant qlora_ours_v4_mix_relca_v2_Qwen2.5-72B-Instruct_description_Llama-3.2-90B-Vision-Instruct_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca_v2 --wandb --description Llama-3.2-90B-Vision-Instruct
```

</details>