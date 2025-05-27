import argparse
import json


def extract_tofu_resluts(path):
    with open(path, "r") as file:
        data = json.load(file)
    forget_truth_ratio = data["forget"]["truth_ratio"]
    forget_truth_prob = data["forget"]["truth_prob"]
    forget_rougeL_score = data["forget"]["rougeL_score"]
    forget_acc = data["forget"]["acc"]
    retain_truth_ratio = data["retain"]["truth_ratio"]
    retain_truth_prob = data["retain"]["truth_prob"]
    retain_rougeL_score = data["retain"]["rougeL_score"]
    retain_acc = data["retain"]["acc"]
    real_author_truth_ratio = data["real_author"]["truth_ratio"]
    real_author_truth_prob = data["real_author"]["truth_prob"]
    real_author_rougeL_score = data["real_author"]["rougeL_score"]
    real_author_acc = data["real_author"]["acc"]
    world_fact_truth_ratio = data["world_fact"]["truth_ratio"]
    world_fact_truth_prob = data["world_fact"]["truth_prob"]
    world_fact_rougeL_score = data["world_fact"]["rougeL_score"]
    world_fact_acc = data["world_fact"]["acc"]
    forget_quality = data["Forget Quality"]
    MIA = data["MIA"]["Min_50.0% Prob"]
    return (
        forget_truth_ratio,
        forget_truth_prob,
        forget_rougeL_score,
        forget_acc,
        retain_truth_ratio,
        retain_truth_prob,
        retain_rougeL_score,
        retain_acc,
        real_author_truth_ratio,
        real_author_truth_prob,
        real_author_rougeL_score,
        real_author_acc,
        world_fact_truth_ratio,
        world_fact_truth_prob,
        world_fact_rougeL_score,
        world_fact_acc,
        forget_quality,
        MIA
    )


def extract_few_shots_results(path):
    with open(path, "r") as file:
        data = json.load(file)
    # Extract the specified tasks and their accuracies
    tasks = [
        "arc_challenge",
        "arc_easy",
        "boolq",
        "hellaswag",
        "openbookqa",
        "piqa",
        "rte",
        "winogrande",
    ]
    acc_values = [
        data["results"][task]["acc,none"] for task in tasks if task in data["results"]
    ]
    average_acc = sum(acc_values) / len(acc_values) if acc_values else None
    return average_acc, acc_values


def extract_truthfulqa_results(path):
    with open(path, "r") as file:
        data = json.load(file)
    bleu_diff = data["results"]["truthfulqa_gen"]["bleu_diff,none"]
    rouge1_diff = data["results"]["truthfulqa_gen"]["rouge1_diff,none"]
    mc1 = data["results"]["truthfulqa_mc1"]["acc,none"]
    mc2 = data["results"]["truthfulqa_mc2"]["acc,none"]
    return bleu_diff, rouge1_diff, mc1, mc2


def extract_ppl_results(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data["results"]["wikitext"]["word_perplexity,none"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract results for Detoxify unlearning")
    parser.add_argument("--root", type=str, help="path to the root directory")
    args = parser.parse_args()
    few_shots_path = f"{args.root}/few_shots.json"
    # forget_path = f"{args.root}/forget.json"
    tofu_path = f"{args.root}/tofu.json"
    harmful_path = f"{args.root}/harmful.json"
    ppl_path = f"{args.root}/ppl.json"
    results = []
    (
        forget_truth_ratio,
        forget_truth_prob,
        forget_rougeL_score,
        forget_acc,
        retain_truth_ratio,
        retain_truth_prob,
        retain_rougeL_score,
        retain_acc,
        real_author_truth_ratio,
        real_author_truth_prob,
        real_author_rougeL_score,
        real_author_acc,
        world_fact_truth_ratio,
        world_fact_truth_prob,
        world_fact_rougeL_score,
        world_fact_acc,
        forget_quality,
        MIA
    ) = extract_tofu_resluts(tofu_path)
    # few_shots_average_acc, few_shots_acc_values = extract_few_shots_results(
    #     few_shots_path
    # )
    # bleu_diff, rouge1_diff, mc1, mc2 = extract_truthfulqa_results(few_shots_path)
    # ppl_value = extract_ppl_results(ppl_path)
    # results.append(forget_truth_ratio)
    # results.append(forget_truth_prob)
    # results.append(forget_rougeL_score)
    # results.append(retain_truth_ratio)
    # results.append(retain_truth_prob)
    # results.append(retain_rougeL_score)
    # results.append(real_author_truth_ratio)
    # results.append(real_author_truth_prob)
    # results.append(real_author_rougeL_score)
    # results.append(world_fact_truth_ratio)
    # results.append(world_fact_truth_prob)
    # results.append(world_fact_rougeL_score)
    # results.append(ppl_value)
    # results.append(few_shots_average_acc)
    # results.append(mc1)
    # results.append(mc2)
    results.append(forget_quality)
    results.append(forget_acc)
    results.append(forget_rougeL_score)
    results.append(MIA)
    results.append(retain_acc)
    results.append(retain_rougeL_score)
    results.append(real_author_acc)
    results.append(real_author_rougeL_score)
    results.append(world_fact_acc)
    results.append(world_fact_rougeL_score)

    print(",".join([str(result) for result in results]))