<div align='center'>
 
# Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning


[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/WAGLE?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)

</div>

This is the official code repository for the paper [Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](https://arxiv.org/pdf/2410.17509).

## Abstract

Machine unlearning presents a promising approach to mitigating privacy and safety concerns in large language models (LLMs) by enabling the selective removal of targeted data or knowledge while preserving model utility. However, existing unlearning methods remain over-sensitive to downstream fine-tuning, which can rapidly recover what is supposed to be unlearned information even when the fine-tuning task is entirely {unrelated} to the unlearning objective.
To enhance robustness, we introduce the concept of `invariance' into unlearning for the first time from the perspective of invariant risk minimization (IRM), a principle for environment-agnostic training. By leveraging IRM, we develop a new invariance-regularized LLM unlearning framework, termed invariant LLM unlearning (ILU). 
We show that the proposed invariance regularization, even using only a single fine-tuning dataset during ILU training, can enable unlearning robustness to generalize effectively across diverse and new fine-tuning tasks at test time. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP benchmark, which focuses on removing an LLM's hazardous knowledge generation capabilities, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.

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


