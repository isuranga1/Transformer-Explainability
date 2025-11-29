import os
import subprocess
import re

METHOD = "ours"   # set this! e.g., "ours"
BASE_DIR = f"bert_models/movies/{METHOD}"
DATA_DIR = "BertData/data/movies/"
SPLIT = "test"

# Output directory
f1_dir = os.path.join(BASE_DIR, "f1-scores")
os.makedirs(f1_dir, exist_ok=True)

# -------------------------------------
# 1. Detect which k files exist
# -------------------------------------
k_values = []

for filename in os.listdir(BASE_DIR):
    match = re.match(r"identifier_results_(\d+)\.json$", filename)
    if match:
        k_values.append(int(match.group(1)))

k_values.sort()

if not k_values:
    print("❌ No identifier_results_k.json files found.")
    exit(1)

print("Detected ks:", k_values)

# -------------------------------------
# 2. Execute metrics.py for each k
# -------------------------------------
for k in k_values:
    input_file = os.path.join(BASE_DIR, f"identifier_results_{k}.json")
    output_file = os.path.join(f1_dir, f"top-{k}.json")

    print(f"\n▶ Processing K={k}")

    cmd = [
        "python3",
        "BERT_rationale_benchmark/metrics.py",
        "--data_dir", DATA_DIR,
        "--split", SPLIT,
        "--results", input_file,
        "--score_file", output_file,
    ]

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode != 0:
        print(f"❌ Error running metrics.py for k={k}")
        print(result.stderr)
        continue

    print(f"✔ Scores saved to {output_file}")

print("\n🎉 Done!")
