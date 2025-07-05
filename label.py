import os
import pandas as pd 
from scipy.stats import mannwhitneyu

doc = # (file path)

for file in os.listdir(doc):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(doc, file))
        model_name = file.split("_")[0]
        
        uuoo = df["uuoo"].dropna().values
        uoou = df["uoou"].dropna().values
        oouu = df["oouu"].dropna().values
        ouuo = df["ouuo"].dropna().values
        
        stat_uuoo_uoou, p_uuoo_uoou = mannwhitneyu(uuoo, uoou, alternative='two-sided')
        stat_uuoo_oouu, p_uuoo_oouu = mannwhitneyu(uuoo, oouu, alternative='two-sided')
        stat_uuoo_ouuo, p_uuoo_ouuo = mannwhitneyu(uuoo, ouuo, alternative='two-sided')
        stat_uoou_oouu, p_uoou_oouu = mannwhitneyu(uoou, oouu, alternative='two-sided')
        stat_uoou_ouuo, p_uoou_ouuo = mannwhitneyu(uoou, ouuo, alternative='two-sided')
        stat_oouu_ouuo, p_oouu_ouuo = mannwhitneyu(oouu, ouuo, alternative='two-sided')

        print(f"{model_name} - uuoo vs uoou: U={stat_uuoo_uoou:.4f}, p={p_uuoo_uoou:.4e} - {'Significant' if p_uuoo_uoou < 0.05 else 'Not significant'}")
        print(f"{model_name} - uuoo vs oouu: U={stat_uuoo_oouu:.4f}, p={p_uuoo_oouu:.4e} - {'Significant' if p_uuoo_oouu < 0.05 else 'Not significant'}")
        print(f"{model_name} - uuoo vs ouuo: U={stat_uuoo_ouuo:.4f}, p={p_uuoo_ouuo:.4e} - {'Significant' if p_uuoo_ouuo < 0.05 else 'Not significant'}")
        print(f"{model_name} - uoou vs oouu: U={stat_uoou_oouu:.4f}, p={p_uoou_oouu:.4e} - {'Significant' if p_uoou_oouu < 0.05 else 'Not significant'}")
        print(f"{model_name} - uoou vs ouuo: U={stat_uoou_ouuo:.4f}, p={p_uoou_ouuo:.4e} - {'Significant' if p_uoou_ouuo < 0.05 else 'Not significant'}")
        print(f"{model_name} - oouu vs ouuo: U={stat_oouu_ouuo:.4f}, p={p_oouu_ouuo:.4e} - {'Significant' if p_oouu_ouuo < 0.05 else 'Not significant'}")


