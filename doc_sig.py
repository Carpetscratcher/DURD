from scipy.stats import mannwhitneyu
import pandas as pd
import os
import csv
import math
import itertools

folder_cctns = "C:/Users/alvin_xtd9mn0/Documents/scriptie/cctns"
folder_gamma_j = "C:/Users/alvin_xtd9mn0/Documents/scriptie/gamma_j"
folder_gemini = "C:/Users/alvin_xtd9mn0/Documents/scriptie/gemini"
folder_qheadache = "C:/Users/alvin_xtd9mn0/Documents/scriptie/qheadache"
folder_tcs = "C:/Users/alvin_xtd9mn0/Documents/scriptie/tcs"
folder_themas = "C:/Users/alvin_xtd9mn0/Documents/scriptie/themas"

for file in os.listdir(folder_cctns):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_cctns, file))
        cctns_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

for file in os.listdir(folder_gamma_j):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_gamma_j, file))
        gamma_j_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

for file in os.listdir(folder_gemini):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_gemini, file))
        gemini_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

for file in os.listdir(folder_qheadache):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_qheadache, file))
        qheadache_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

for file in os.listdir(folder_tcs):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_tcs, file))
        tcs_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

for file in os.listdir(folder_themas):
    if file.endswith("_raw_perplexities.csv"):
        df = pd.read_csv(os.path.join(folder_themas, file))
        themas_perplexities = df[['uuoo', 'uoou', 'oouu', 'ouuo']].values.flatten().tolist()

docs = {
    "cctns": cctns_perplexities,
    "gamma_j": gamma_j_perplexities,
    "gemini": gemini_perplexities,
    "qheadache": qheadache_perplexities,
    "tcs": tcs_perplexities,
    "themas": themas_perplexities
}
# Pairwise Mann-Whitney U test
for (doc1, values1), (doc2, values2) in itertools.combinations(docs.items(), 2):
    stat, p = mannwhitneyu(values1, values2, alternative='two-sided')
    
    if p < 0.05:
        a = 0
        print(f"{doc1} vs {doc2}: U={stat:.4f}, p={p:.4e}")
        print(f"  Significant difference found between {doc1} and {doc2}")
    else:
        a = 0
        # print(f"{doc1} vs {doc2}: U={stat:.4f}, p={p:.4e}")
        # print(f"  No significant difference between {doc1} and {doc2}")


