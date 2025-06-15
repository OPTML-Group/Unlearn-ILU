<div align='center'>
 
# Reasoning Model Unlearning: Forgetting Traces, Not Just Answers,  While Preserving Reasoning Skills


[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/WAGLE?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)

</div>

This is the official code repository for the paper [Reasoning Model Unlearning: Forgetting Traces, Not Just Answers,  While Preserving Reasoning Skills]().

## Abstract

Recent advances in large reasoning models (LRMs) have enabled strong chain-of-thought (CoT) generation through test-time computation. While these multi-step reasoning capabilities represent a major milestone in language model performance, they also introduce new safety risks. In this work, we present the first systematic study to revisit the problem of \textit{machine unlearning in the context of LRMs}. Machine unlearning refers to the process of removing the influence of sensitive, harmful, or undesired data or knowledge from a trained model without full retraining. We show that conventional unlearning algorithms, originally designed for non-reasoning models, are inadequate for LRMs. In particular, even when final answers are successfully erased, sensitive information often persists within the intermediate reasoning steps, \textit{i.e.}, CoT trajectories.
 To address this challenge, we extend conventional unlearning and propose \underline{R}easoning-aware \underline{R}epresentation \underline{M}isdirection for \underline{U}nlearning (\textbf{\ours{}}), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model’s reasoning ability.
 Our experiments demonstrate that {\ours} significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B.

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
conda create -n ILU python=3.9
conda activate ILU
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install datasets wandb transformers==4.37.2 sentencepiece sentence-transformers==2.6.1
pip install git+https://github.com/changsheng/fastargs  
pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn
cd lm-evaluation-harness
pip install -e .
```



## Code structure

```
-- configs/: Contains the configuration files for the experiments.
    -- Different folders for different experiments (MUSE, WMDP, etc.)
-- files/: 
    -- data/: Contains the data files necessary for the experiments.
    -- results/: the log and results of experiments will stored in this directory.
-- lm-evaluation-harness: official repository for the evaluation of LLMs from      
  https://github.com/EleutherAI/lm-evaluation-harness.
-- src/: Contains the source code for the experiments.
    -- dataset/: Contains the data processing and dataloader creation codes.
    -- model/: Contains the main unlearning class which will conduct load model, 
      unlearn,evaluation.
    -- optim/: Contains the optimizer code.
    -- metrics/: Contains the evaluation code.
    -- loggers/: Contains the logger code.
    -- unlearn/: Contains different unlearning methods' code also mask generation code.
    -- exec/:
        -- Fine_tune_hp.py: Code for finetuning on harry potter books.
        -- unlearn_model.py: The main file to run the unlearning experiments.
```
## Running the experiments

First, you need to download the mask files from Google Drive and place them into ```./mask/``` directory. You can download the mask files from [here](https://drive.google.com/drive/folders/1yYzvroNHNKWrNWk0WOX_j4pXw4kIyEyf?usp=sharing). Those mask files name should be like ```{task_name}_{ratio}.pt```. For example, ```tofu_0.8.pt``` represents the mask file for TOFU task with 80% weights are selected for unlearning from WAGLE method.

After downloading the mask files, you can run the following command to run the experiments:
```
python src/exec/unlearn_model.py --config_file configs/{unlearn_task}/{unlearn_method}.json --unlearn.mask_path mask/{unlearn_task}_{ratio}.pt {other_args}
```


