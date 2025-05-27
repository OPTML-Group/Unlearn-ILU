import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

# Set up argparse to accept command line arguments for the mask file paths
parser = argparse.ArgumentParser(description="Compare two model masks for similarity.")
parser.add_argument("--mask_dir", type=str, help="Path to the joint mask file.")
parser.add_argument("--max_score", type=float, default=1.0, help="Maximum score")
args = parser.parse_args()

plt.style.use('ggplot') 
# Load the scores from file
scores = torch.load(f"{args.mask_dir}/scores.pt")
scores = -scores.type(torch.float16)
scores = scores.numpy()
weights = np.ones_like(scores) / len(scores)
# Plot the distribution of scores
plt.figure(figsize=(10, 6))
plt.hist(
    scores,
    bins=100,
    alpha=0.7,
    color="blue",
    range=(0, args.max_score),
    weights=weights,
    edgecolor="black",
)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.tight_layout()
plt.xlabel("Score", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.savefig(f"{args.mask_dir}/scores_distribution.png")
plt.close()

import numpy as np
import matplotlib.pyplot as plt

for percent in [20, 60, 80, 40]:
    # Calculate the percentile as the threshold
    threshold = np.percentile(scores, percent)
    print(f"{percent}th Percentile Threshold: {threshold}")
    plt.hist(scores, bins=50, alpha=0.7, label='Scores Distribution',range=(0, args.max_score), weights=weights)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.axvline(threshold, color='r', linestyle='--', label=f'{percent}th Percentile Threshold')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{args.mask_dir}/scores_distribution_{percent}th_percentile.png")
    plt.close()

