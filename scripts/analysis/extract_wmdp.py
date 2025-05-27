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
        "mmlu",
    ]
    acc_values = [
        data["results"][task]["acc,none"] for task in tasks if task in data["results"]
    ]
    average_acc = sum(acc_values) / len(acc_values) if acc_values else None
    return average_acc, acc_values

def extract_wmdp_results(path):
    with open(path, "r") as file:
        data = json.load(file)
    tasks = ["wmdp","wmdp_bio","wmdp_chem","wmdp_cyber"]
    acc_values = [
        data["results"][task]["acc,none"] for task in tasks if task in data["results"]
    ]
    wmdp_acc = acc_values[0]
    wmdp_bio_acc = acc_values[1]
    wmdp_chem_acc = acc_values[2]
    wmdp_cyber_acc = acc_values[3]
    return wmdp_acc, wmdp_bio_acc, wmdp_chem_acc, wmdp_cyber_acc




if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract results for Detoxify unlearning")
    parser.add_argument("--root", type=str, help="path to the root directory")
    args = parser.parse_args()
    few_shots_path = f"{args.root}/mmlu.json"
    # forget_path = f"{args.root}/forget.json"
    wmdp_path = f"{args.root}/wmdp.json"
    ppl_path = f"{args.root}/ppl.json"
    results = []
    mmlu,_ = extract_few_shots_results(few_shots_path)
    wmdp_acc, wmdp_bio_acc, wmdp_chem_acc, wmdp_cyber_acc = extract_wmdp_results(wmdp_path)
    results.append(wmdp_bio_acc)
    results.append(wmdp_cyber_acc)
    # results.append(wmdp_chem_acc)
    results.append((wmdp_bio_acc+wmdp_cyber_acc)/2)
    results.append(mmlu)


    print(",".join([str(result) for result in results]))
