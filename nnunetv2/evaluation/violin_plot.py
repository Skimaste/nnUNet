import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
json_path = "/mnt/processing/emil/nnUNet_results/Dataset003_ImageCAS_split/nnUNetTrainerDropout__p02_s3__3d_fullres/inference/evaluation_summary_masked.json"  # ← Update this
output_plot_path = os.path.join(os.path.dirname(json_path), "ece_masked_distribution_violin.png")

# === LOAD JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

results = data.get("results", [])

# === EXTRACT ECE VALUES ===
ece_mean = [r["ECE for probabilities"] for r in results]
ece_var = [r["ECE for variance"] for r in results]
ece_shannon = [r["ECE for entropy"] for r in results]

# === PREPARE DATAFRAME ===
df = pd.DataFrame({
    "ECE": ece_mean + ece_var + ece_shannon,
    "Metric": (["ECE for Probability"] * len(ece_mean)) +
              (["ECE for Variance"] * len(ece_var)) +
              (["ECE for Entropy"] * len(ece_shannon))
})

# === PLOT VIOLIN ===
plt.figure(figsize=(10, 6))
sns.violinplot(x="Metric", y="ECE", data=df, inner="box", scale="width", palette="Set2")
plt.title("ECE (masked) Distribution Across Cases")
plt.ylabel("Expected Calibration Error (ECE)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# === SAVE ===
plt.savefig(output_plot_path)
plt.close()
print(f"Violin plot saved to: {output_plot_path}")


