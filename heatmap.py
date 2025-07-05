import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

folder_path = # (folder path)
document_name = # (doc name)

# verkrijgen van modellen en hun data
model_data = {}
for file in os.listdir(folder_path):
    if file.endswith("_raw_perplexities.csv") and document_name in file:
        model_name = file.split("_")[0]
        df = pd.read_csv(os.path.join(folder_path, file))
        model_data[model_name] = df
        # print(model_data)

prompt_types = ["uuoo", "uoou", "oouu", "ouuo"]

# Kruskal-Wallis test uitvoeren voor elke prompt type
for prompt in prompt_types:
    print(f"Kruskal test voor: {prompt}")

    # Verzamel de perplexity data voor de opgegeven prompt
    rows = []
    for model, df in model_data.items():
        rows.append(df[prompt].dropna().values)

    stat, p = kruskal(*rows)
    print(f"Kruskal-Wallis H = {stat:.4f}, p = {p:.4g}")
    if p < 0.05:
        print("Significant")
    else:
        print("Not significant")

posthoc_matrices = []

for prompt in prompt_types:
    print(f"Dunn's post-hoc test: {prompt}")

    values = []
    labels = []

    for model, df in model_data.items():
        vals = df[prompt].values
        values.extend(vals)
        labels.extend([model] * len(vals))

    df_posthoc = pd.DataFrame({
        "model": labels,
        "perplexity": values
    })

    # Dunn's post hoc test
    p_matrix = sp.posthoc_dunn(df_posthoc, val_col="perplexity", group_col="model")
    print(p_matrix)

    posthoc_matrices.append(p_matrix)

def format_pval(x):
    return f"{x:.1e}" if x < 0.001 else f"{x:.3f}"

avg_matrix = sum(posthoc_matrices) / len(posthoc_matrices)
avg_matrix.to_csv(f"dunn_posthoc_avg_{document_name}.csv")

significance_mask = avg_matrix < 0.05

color_matrix = np.select(
    [avg_matrix == 1.0, avg_matrix >= 0.05, avg_matrix < 0.05],
    [0, 1, 2]
)

cmap = ["lightgray", "#FC9272", "#6BAED6"]
formatted = np.vectorize(format_pval)(avg_matrix.values)

sns.heatmap(color_matrix,
            cmap=cmap,
            annot=formatted,
            fmt="",
            xticklabels=avg_matrix.columns,
            yticklabels=avg_matrix.index,
            cbar=False,
            linewidths=0.5)

plt.title(f"Dunn’s test — gemiddelde p-waardes ({document_name})")
plt.tight_layout()
plt.savefig(f"dunn_heatmap_avg_{document_name}.png")
plt.show()
