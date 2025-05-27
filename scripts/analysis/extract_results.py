import argparse
import json
import os
import sys


# Function to extract, format data for the Markdown table and calculate mean accuracy
def format_to_markdown_with_mean_acc(data):
    # Initialize the table with column names
    markdown_table = "| Test |"

    # Find the test names, accuracies and perplexity
    test_names = []
    accuracies = []
    ppl = None
    for entry in data:
        if "few_shots" in entry:
            test_names = list(entry["few_shots"]["results"].keys())
            ppl = entry.get("ppl", "N/A")
            for test_name in test_names:
                accuracies.append(entry["few_shots"]["results"][test_name]["acc"])
            break

    # Add the test names and perplexity as column names
    for test_name in test_names:
        markdown_table += f" {test_name} |"
    markdown_table += f" | Mean Accuracy | Perplexity \n"

    # Add separators for the columns
    markdown_table += "|" + "------|" * (len(test_names) + 2) + "\n"

    # Add a single row of data
    markdown_table += "| Accuracy |"
    for acc in accuracies:
        markdown_table += f" {acc:.2f} |"
    mean_accuracy = sum(accuracies) / len(accuracies)
    markdown_table += f" {mean_accuracy:.2f} | {ppl} |\n"

    return markdown_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    data = json.load(open(os.path.join(args.root, "log.json"), "r"))
    # Create the markdown table
    markdown_output_with_mean_acc = format_to_markdown_with_mean_acc(data)
    # save the markdown table
    with open(os.path.join(args.root, "log.md"), "w") as f:
        f.write(markdown_output_with_mean_acc)
    print(markdown_output_with_mean_acc)
