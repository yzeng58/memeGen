# Benchmarking Meme Humor with Large Language Models

## Baseline Performance

### Single Meme Evaluation

```bash
# ours_v3
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb

# Relca
python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb
```

### Pairwise Meme Comparison

```bash
# ours_v3 
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb

# relca
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb

```

<details>
<summary>Click to expand full experiments</summary>

### Pairwise Meme Comparison

#### Qwen-VL
```bash
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct
```

#### GPT
```bash
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o-mini
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o
```

#### Pixtral
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name pixtral-12b 

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name pixtral-12b 
```

#### Gemini
```bash
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
# TODO
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro
```

### Single Meme Evaluation 
#### Qwen-VL
```bash
python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-2B-Instruct

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-7B-Instruct

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct


# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-2B-Instruct

# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-7B-Instruct

# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct
```

#### GPT
```bash
python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o-mini

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o

# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o-mini
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o-mini

# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o
```

#### Pixtral
```bash
# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b
```

#### Gemini
```bash
# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-flash

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-flash

# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro 
```

#### LLama-3.2
```bash
# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-11B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-11B-Vision-Instruct

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-11B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-11B-Vision-Instruct


# TODO
python evaluation.py --dataset_name ours_v3 --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct

python evaluation.py --dataset_name relca --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct
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

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3
```

##### Llama-3.2-90B-Vision-Instruct
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v3
```

##### Pixtral-12b
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name pixtral-12b --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v3
```

##### GPT-4o
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gpt-4o --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v3
```

##### gemini-1.5-pro
```bash
# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name relca
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca

# TODO
python evaluation.py --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name gemini-1.5-pro --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v3
```

##### Mixtral/Mistral
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mixtral-8x22B-Instruct-v0.1 --dataset_name ours_v3

python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-Large-Instruct-2407 --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-Large-Instruct-2407 --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Mistral-Large-Instruct-2407 --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Mistral-Large-Instruct-2407 --dataset_name ours_v3
```

##### Llama-3.1 
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-70B-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Llama-3.1-70B-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Llama-3.1-70B-Instruct --dataset_name ours_v3
```

##### Qwen2.5
```bash
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-72B-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name relca

# TODO
python evaluation.py --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode single --n_demos 2 --model_name Qwen2.5-72B-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --prompt_name single --n_demos 2 --not_load_model --wandb --model_name Qwen2.5-72B-Instruct --dataset_name ours_v3
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
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2-VL-72B-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2-VL-72B-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Qwen2-VL-72B-Instruct --dataset_name ours_v3
```

##### Llama-3.2-90B-Vision-Instruct
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name Llama-3.2-90B-Vision-Instruct --dataset_name ours_v3
```

##### Pixtral-12b
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name pixtral-12b --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name pixtral-12b --dataset_name relca

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name pixtral-12b --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name pixtral-12b --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name pixtral-12b --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v3

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name pixtral-12b --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name pixtral-12b --dataset_name ours_v3
```

##### GPT-4o
```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gpt-4o --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gpt-4o --dataset_name relca

python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gpt-4o --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gpt-4o --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gpt-4o --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v3

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gpt-4o --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gpt-4o --dataset_name ours_v3
```

##### gemini-1.5-pro
```bash
# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gemini-1.5-pro --dataset_name relca
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gemini-1.5-pro --dataset_name relca   
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name relca

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 4 --model_name gemini-1.5-pro --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v3

# TODO
python evaluation.py --data_mode test --eval_mode single --n_demos 6 --model_name gemini-1.5-pro --dataset_name ours_v3
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 6  --not_load_model --wandb --model_name gemini-1.5-pro --dataset_name ours_v3
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
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v3 --model_name Qwen2-VL-72B-Instruct
```

##### Llama-3.2-90B-Vision-Instruct
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca --model_name Llama-3.2-90B-Vision-Instruct

# TODO
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v3 --model_name Llama-3.2-90B-Vision-Instruct
```

##### GPT-4o
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca --model_name gpt-4o

# TODO
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v3 --model_name gpt-4o
```

##### Gemini-1.5-Pro
```bash
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name relca --model_name gemini-1.5-pro

# TODO
python evaluation.py --data_mode test --eval_mode pairwise --prompt_name theory --theory_version v4 --data_mode both --n_pairs 2000 --train_ml_model xgboost --wandb --dataset_name ours_v3 --model_name gemini-1.5-pro
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
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct--data_mode test --eval_mode pairwise --wandb --dataset_name ours_v3 --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name Qwen2-VL-72B-Instruct
```


##### Gemini-1.5-Pro
```bash
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca --model_name gemini-1.5-pro

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v3 --model_name gemini-1.5-pro

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name gemini-1.5-pro
```

##### GPT-4o
```bash
# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --context Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name ours_v3 --model_name gpt-4o

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name gpt-4o
```

##### Mixtral-8x22B-Instruct-v0.1
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca --model_name Mixtral-8x22B-Instruct-v0.1

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name Mixtral-8x22B-Instruct-v0.1

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name Mixtral-8x22B-Instruct-v0.1 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name Mixtral-8x22B-Instruct-v0.1
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --wandb --dataset_name relca --model_name Llama-3.1-70B-Instruct

python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct  --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name relca --model_name Llama-3.1-70B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name Llama-3.1-70B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --description Llama-3.2-90B-Vision-Instruct  --data_mode test --eval_mode pairwise --prompt_name cot --wandb --dataset_name ours_v3 --model_name Llama-3.1-70B-Instruct
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
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v3 --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca --model_name Qwen2-VL-72B-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v3 --model_name Qwen2-VL-72B-Instruct
```

##### Llama-3.1-70B-Instruct
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v3 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v3 --model_name Llama-3.1-70B-Instruct --description Llama-3.2-90B-Vision-Instruct
```

##### Mixtral-8x22B-Instruct-v0.1
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v3 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v3 --model_name Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct
```

##### GPT-4o
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name relca --model_name gpt-4o 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 2 --wandb --dataset_name ours_v3 --model_name gpt-4o 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca --model_name gpt-4o 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v3 --model_name gpt-4o 
```

##### Gemini-1.5-Pro
```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name relca --model_name gemini-1.5-pro 

# TODO
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb --dataset_name ours_v3 --model_name gemini-1.5-pro 
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
python finetune.py --eval_mode single --dataset_name relca ours_v3 
python evaluation.py --eval_mode single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 


python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb

```

<details>
<summary>Click to expand full experiments</summary>

###### Qwen2-VL-72B-Instruct
```bash
python finetune.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
# TODO
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb


python finetune.py --eval_mode single --dataset_name relca ours_v3  --model_name Qwen2-VL-72B-Instruct
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca 
python evaluation.py --eval_mode single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 
python evaluation.py --prompt_name single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-72B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb
```

##### Llama-3.2-90B-Vision-Instruct
```bash
python finetune.py --eval_mode single --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca
python evaluation.py --eval_mode single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --model_name Llama-3.2-90B-Vision-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
# TODO
python evaluation.py --eval_mode single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3
python evaluation.py --model_name Llama-3.2-90B-Vision-Instruct --eval_mode pairwise --prompt_name single --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb

python finetune.py --eval_mode single --dataset_name relca ours_v3 --model_name Llama-3.2-90B-Vision-Instruct
python evaluation.py --eval_mode single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 
python evaluation.py --eval_mode single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca
python evaluation.py --prompt_name single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb
```
</details>

##### Llama-3.1-70B-Instruct
```bash
# TODO
```

### Pairwise Meme Comparison 

#### Standard 

```bash
# one dataset
python finetune.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct
python evaluation.py --model_name Qwen2-VL-2B-Instruct --eval_mode pairwise --peft_variant qlora_relca_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb

# mixture
python finetune.py --eval_mode pairwise --dataset_name relca ours_v3 
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --wandb
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --wandb


python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Qwen2-VL-2B-Instruct --peft_variant qlora_relca_mix_ours_v3_Qwen2-VL-2B-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb
```

<details>
<summary>Click to expand full experiments</summary>

###### Qwen2-VL-72B-Instruct
```bash
python finetune.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --dataset_name relca
python evaluation.py --model_name Qwen2-VL-72B-Instruct --eval_mode pairwise --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca
# TODO
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_relca_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --wandb

python finetune.py --eval_mode pairwise --dataset_name ours_v3 relca --model_name Qwen2-VL-72B-Instruct
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_ours_v3_mix_relca_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --wandb
python evaluation.py --eval_mode pairwise --model_name Qwen2-VL-72B-Instruct --peft_variant qlora_ours_v3_mix_relca_Qwen2-VL-72B-Instruct_multimodal_pairwise_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --wandb
```

# NEED TO BE DEBUGGED
###### Llama-3.2-90B-Vision-Instruct
```bash
python finetune.py --eval_mode pairwise --model_name Llama-3.2-90B-Vision-Instruct --dataset_name relca
python evaluation.py --model_name Llama-3.2-90B-Vision-Instruct --eval_mode pairwise --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --wandb --dataset_name relca
# TODO
python evaluation.py --eval_mode pairwise --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --wandb

python finetune.py --eval_mode pairwise --dataset_name relca ours_v3 
python evaluation.py --eval_mode pairwise --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --wandb
python evaluation.py --eval_mode pairwise --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --wandb
python evaluation.py --prompt_name single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name relca --not_load_model --wandb
python evaluation.py --prompt_name single --model_name Llama-3.2-90B-Vision-Instruct --peft_variant qlora_relca_mix_ours_v3_Llama-3.2-90B-Vision-Instruct_multimodal_single_standard_0_shot_train --n_pairs 2000 --data_mode test --dataset_name ours_v3 --not_load_model --wandb
```

##### Llama-3.1-70B-Instruct
```bash
# TODO
```
</details>
