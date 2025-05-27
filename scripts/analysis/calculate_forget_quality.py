from scipy.stats import ks_2samp
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract results for Detoxify unlearning")
    parser.add_argument("--path", type=str, help="path to the forget path")
    parser.add_argument("--retrain_path", type=str, default="storage/Results/Retrain/tofu.json")    

    args = parser.parse_args()

    with open(args.path, "r") as file:
        forget_data = json.load(file)
    
    with open(args.retrain_path, "r") as file:
        retrain_data = json.load(file)

    forget_truth_ratios = forget_data["truth_ratios"]
    retain_truth_ratios = retrain_data["truth_ratios"]

    test_res = ks_2samp(forget_truth_ratios, retain_truth_ratios)

    print(test_res.pvalue)

