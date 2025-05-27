import argparse
import json


def extract_PII_resluts(path):
    with open(path, "r") as file:
        data = json.load(file)
    context_acc = data["context_acc"]
    one_shot_non_domain_acc = max(data["one_shot_non_domain_acc"].values())
    one_shot_acc = max(data["one_shot_acc"].values())
    two_shot_non_domain_acc = max(data["two_shot_non_domain_acc"].values())
    two_shot_acc = max(data["two_shot_acc"].values())
    return (
        context_acc,
        one_shot_non_domain_acc,
        one_shot_acc,
        two_shot_non_domain_acc,
        two_shot_acc,
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
    PII_path = f"{args.root}/PII.json"
    harmful_path = f"{args.root}/harmful.json"
    ppl_path = f"{args.root}/ppl.json"
    results = []
    (
        context_acc,
        one_shot_non_domain_acc,
        one_shot_acc,
        two_shot_non_domain_acc,
        two_shot_acc,
    ) = extract_PII_resluts(PII_path)
    few_shots_average_acc, few_shots_acc_values = extract_few_shots_results(
        few_shots_path
    )
    bleu_diff, rouge1_diff, mc1, mc2 = extract_truthfulqa_results(few_shots_path)
    ppl_value = extract_ppl_results(ppl_path)
    results.append(context_acc)
    results.append(one_shot_acc)
    results.append(two_shot_acc)
    results.append(one_shot_non_domain_acc)
    results.append(two_shot_non_domain_acc)
    results.append(ppl_value)
    results.append(few_shots_average_acc)
    results.append(mc1)
    results.append(mc2)
    results.append(bleu_diff)
    results.append(rouge1_diff)

    print(",".join([str(result) for result in results]))
