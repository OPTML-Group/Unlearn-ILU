<div align='center'>
 
# WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models

[![preprint](https://img.shields.io/badge/arXiv-2410.17509-B31B1B)](https://arxiv.org/pdf/2410.17509)

[![Venue:NeurIPS 2024](https://img.shields.io/badge/Venue-NeurIPS%202024-blue)](https://neurips.cc/Conferences/2024)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/WAGLE?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/WAGLE)](https://github.com/OPTML-Group/WAGLE)

</div>

This is the official code repository for the paper [WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models](https://arxiv.org/pdf/2410.17509).

## Abstract

The need for effective unlearning mechanisms in large language models (LLMs) is increasingly urgent, driven by the necessity to adhere to data regulations and foster ethical generative AI practices. LLM unlearning is designed to reduce the impact of undesirable data influences and associated model capabilities without diminishing the utility of the model if unrelated to the information being forgotten. Despite growing interest, much of the existing research has focused on varied unlearning method designs to boost effectiveness and efficiency. However, the inherent relationship between model weights and LLM unlearning has not been extensively examined. In this paper, we systematically explore how model weights interact with unlearning processes in LLMs and we design the weight attribution-guided LLM unlearning method, WAGLE, which unveils the interconnections between 'influence' of weights and 'influence' of data to forget and retain in LLM generation. By strategically guiding the LLM unlearning across different types of unlearning methods and tasks, WAGLE can erase the undesired content, while maintaining the performance of the original tasks. We refer to the weight attribution-guided LLM unlearning method as WAGLE, which unveils the interconnections between 'influence' of weights and 'influence' of data to forget and retain in LLM generation. Our extensive experiments show that WAGLE boosts unlearning performance across a range of LLM unlearning methods such as gradient difference and (negative) preference optimization, applications such as fictitious unlearning (TOFU benchmark), malicious use prevention (WMDP benchmark), and copyrighted information removal, and models including Zephyr-7b-beta and Llama2-7b. To the best of our knowledge, our work offers the first principled method for attributing and pinpointing the influential weights in enhancing LLM unlearning. It stands in contrast to previous methods that lack weight attribution and simpler weight attribution techniques.

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
conda create -n WAGLE python=3.9
conda activate WAGLE
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install datasets wandb transformers==4.37.2 sentencepiece sentence-transformers==2.6.1
pip install git+https://github.com/jinghanjia/fastargs  
pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn
cd lm-evaluation-harness
pip install -e .
```

## WMDP Unlearned Models
Please feel free to use the following models for your research:

WAGLE+GradDiff: [🤗 flyingbugs/WMDP_GradDiff_WAGLE_Zephyr_7B](https://huggingface.co/flyingbugs/WMDP_GradDiff_WAGLE_Zephyr_7B)

WAGLE+NPO: [🤗 flyingbugs/WMDP_NPO_WAGLE_Zephyr_7B](https://huggingface.co/flyingbugs/WMDP_NPO_WAGLE_Zephyr_7B)


## Code structure

```
-- configs/: Contains the configuration files for the experiments.
    -- Different folders for different experiments (Tofu, WMDP, etc.)
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


<!---## Cite This Work
```
@misc{jia2024waglestrategicweightattribution,
      title={WAGLE: Strategic Weight Attribution for Effective and Modular Unlearning in Large Language Models}, 
      author={Jinghan Jia and Jiancheng Liu and Yihua Zhang and Parikshit Ram and Nathalie Baracaldo and Sijia Liu},
      year={2024},
      eprint={2410.17509},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.17509}, 
}
```!--->
