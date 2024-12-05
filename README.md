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

### Pairwise Meme Comparison Results

#### Qwen-VL
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-2B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-7B-Instruct
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name Qwen2-VL-72B-Instruct
```

#### GPT
```bash
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gpt-4o-mini
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
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name ours_v3 --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro

python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-flash
python evaluation.py --n_pairs 2000 --dataset_name relca --data_mode test --eval_mode pairwise --wandb --model_name gemini-1.5-pro
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

#### In-Context Learning

```bash
python evaluation.py --data_mode test --eval_mode single --n_demos 4
python evaluation.py --n_pairs 2000 --data_mode test --prompt_name single --n_demos 4 --not_load_model --wandb
```

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

#### In-Context Learning

```bash
python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --n_demos 4 --wandb
```

#### Ensemble

```bash
# Example Ensemble -- need to correspondingly adjust description and context parameters 

python evaluation.py --n_pairs 2000 --data_mode test --eval_mode pairwise --model_name Llama-3.1-70B-Instruct Qwen2-VL-72B-Instruct Mixtral-8x22B-Instruct-v0.1 --description Llama-3.2-90B-Vision-Instruct '' Llama-3.2-90B-Vision-Instruct --context '' '' '' --wandb --not_load_model
```

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
