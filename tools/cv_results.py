import os
import re

RESULTS_DIR = "results_cv_5fold_5ep_p_sweep"
P_VALUES = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"]
MODELS = ["index", "random", "mixer"]

STD_PATTERN = re.compile(r"Val\s+accuracy\s+over\s+folds:\s*[0-9.]+%\s*±\s*([0-9.]+)%")


def parse_std(path):
    with open(path, "r") as f:
        txt = f.read()
    match = STD_PATTERN.search(txt)
    return float(match.group(1)) if match else None


def collect_std_results():
    results = {}
    for model in MODELS:
        for p in P_VALUES:
            path = os.path.join(RESULTS_DIR, f"{model}_p{p}.txt")
            if os.path.exists(path):
                std = parse_std(path)
                if std is not None:
                    results[(model, p)] = std
    return results


def print_std_table(results):
    print("\n=== Cross-validation Accuracy Std (Val) ===\n")
    header = f"{'p\'':>5}"
    for model in MODELS:
        header += f" {model.capitalize():>10}"
    print(header)
    print("-" * (6 + 11 * len(MODELS)))
    for p in P_VALUES:
        row = f"{p:>5}"
        for model in MODELS:
            key = (model, p)
            if key in results:
                row += f" {results[key]:10.2f}"
            else:
                row += f" {'-':>10}"
        print(row)


if __name__ == "__main__":
    std_results = collect_std_results()
    print_std_table(std_results)
