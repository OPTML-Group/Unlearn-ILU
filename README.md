<div align='center'>
 
# Reasoning Model Unlearning: Forgetting Traces, Not Just Answers,  While Preserving Reasoning Skills


[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/WAGLE?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)

</div>

This is the official code repository for the paper [Reasoning Model Unlearning: Forgetting Traces, Not Just Answers,  While Preserving Reasoning Skills]().

## Abstract

Recent advances in large reasoning models (LRMs) have enabled strong chain-of-thought (CoT) generation through test-time computation. While these multi-step reasoning capabilities represent a major milestone in language model performance, they also introduce new safety risks. In this work, we present the first systematic study to revisit the problem of machine unlearning in the context of LRMs. Machine unlearning refers to the process of removing the influence of sensitive, harmful, or undesired data or knowledge from a trained model without full retraining. We show that conventional unlearning algorithms, originally designed for non-reasoning models, are inadequate for LRMs. In particular, even when final answers are successfully erased, sensitive information often persists within the intermediate reasoning steps, i.e., CoT trajectories.
 To address this challenge, we extend conventional unlearning and propose **R**easoning-aware **R**epresentation **M** isdirection for **U** unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model’s reasoning ability.
 Our experiments demonstrate that $R^2MU$ significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B.

<!-- <table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Systematic overview and experiment highlights of SimNPO.</em>
    </td>
  </tr>
</table> -->

## Installation

You can install the required dependencies using the following command:
```
conda create -n R2MU python=3.9
conda activate R2MU
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install datasets wandb transformers==4.37.2 sentencepiece sentence-transformers==2.6.1
pip install git+https://github.com/changsheng/fastargs  
pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn
cd lm-evaluation-harness
pip install -e .
```

## Running the experiments

# WMDP
you can run the following command to run the experiments:
```
ALPHA="1.7,1.7"
LR="7.5e-5"
DATA="500"
FILE="v11"
NAME="no_unlearn"
assist_loss="1"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
OUTPUT_DIR="models/${OUTPUT_NAME}"
LOG_FILE="${OUTPUT_NAME}.log"

nohup bash -c "
CUDA_VISIBLE_DEVICES=6,7 python3 -m unlearn_r2mu \
  --model_name_or_path ${MODEL_NAME} \
  --max_num_batches ${DATA} \
  --batch_size 4 \
  --retain_corpora wikitext \
  --forget_corpora original \
  --steering_coeffs 6.5,6.5 \
  --alpha ${ALPHA} \
  --lr ${LR} \
  --assist_loss ${assist_loss} \
  --seed 42 \
  --output_dir ${OUTPUT_DIR} \
  --generated_path generated_all.jsonl \
  --raw_path bio_remove_dataset.jsonl \
  --max_gen_tokens 100 \
  --verbose
" > ${LOG_FILE} 2>&1 &
```
# STAR-1

you can run the following command to run the experiments:
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m rmu.unlearn_safe
 --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
 --max_num_batches 300
 --batch_size 4
 --retain_corpora wikitext
 --forget_corpora safety
 --steering_coeffs 6.5,6.5
 --alpha 20,20
 --lr 1e-06
 --seed 42
 --output_dir models/reasoning_safe_alpha20_lr1p0e-06
 --verbose > logs/reasoning_safe_alpha20_lr1p0e-06.log 2>&1 &

```

