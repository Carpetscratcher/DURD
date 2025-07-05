import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import csv
from scipy.stats import wilcoxon
import os

model_name = 'meta-llama/Llama-2-13b-hf'
# model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'facebook/opt-13b'
# model_name = 'gpt2-xl'
# model_name = 'mistralai/Mistral-7B-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="sequential",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()

# Perplexiteit functie
def calculate_perplexity(inputs):
    perplexities = []
    for prompt in inputs:
        encodings = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexities.append(torch.exp(loss).item())
        del input_ids, outputs
        torch.cuda.empty_cache()
        gc.collect()
    return perplexities

folder_path = # folder path
output_dir = "Llama2-13b" # hernoemen per model
os.makedirs(output_dir, exist_ok=True)
raw_output_dir = "raw_perplexities"
os.makedirs(raw_output_dir, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # laad de CSV in
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL)

        # Genereren van prompts en berekenen van perplexity
        uuoo = []; uoou = []; oouu = []; ouuo = []
        batch_size = 2
        for i in range(math.ceil(len(df)/batch_size)):
            prompts = []
            for j in range(batch_size):
                if (i*batch_size)+j < len(df):
                    s = df['underspecified sentence'].iloc[(i*batch_size)+j]
                    c = df['control sentence'].iloc[(i*batch_size)+j]
                    prompts += [
                        f"This is an underspecified sentence: '{s}'. This is its more specified counterpart: '{c}'.",
                        f"This is an underspecified sentence: '{c}'. This is its more specified counterpart: '{s}'.",
                        f"This is a more specified sentence: '{c}'. This is its underspecified counterpart: '{s}'.",
                        f"This is a more specified sentence: '{s}'. This is its underspecified counterpart: '{c}'."
                    ]
            ppls = calculate_perplexity(prompts)
            for j in range(batch_size):
                if (i*batch_size)+j < len(df):
                    uuoo.append(ppls[j*4])
                    uoou.append(ppls[j*4+1])
                    oouu.append(ppls[j*4+2])
                    ouuo.append(ppls[j*4+3])

        # Save results to dataframe
        df["Mistral over-under uuoo perplexity"] = uuoo
        df["Mistral over-under uoou perplexity"] = uoou
        df["Mistral over-under oouu perplexity"] = oouu
        df["Mistral over-under ouuo perplexity"] = ouuo

        base = os.path.splitext(filename)[0]

        # # Save processed data
        # df.to_csv(f"output_{base}.csv", index=False)

        # --------------------------------------------------------------------------------------- # tekst bestand hernoemen
        with open(os.path.join(output_dir, f"{base}_Llama2-13b.txt"), "w") as f:
            f.write("Average Perplexities:\n")
            f.write(f"uuoo: {np.mean(uuoo):.4f}\n")
            f.write(f"uoou: {np.mean(uoou):.4f}\n")
            f.write(f"oouu: {np.mean(oouu):.4f}\n")
            f.write(f"ouuo: {np.mean(ouuo):.4f}\n")

        pairs = [
            ("uuoo", uuoo, "uoou", uoou),
            ("uuoo", uuoo, "oouu", oouu),
            ("uuoo", uuoo, "ouuo", ouuo),
            ("uoou", uoou, "oouu", oouu),
            ("uoou", uoou, "ouuo", ouuo),
            ("oouu", oouu, "ouuo", ouuo),
        ]

        # -----------------------------------------------------------------------------------------# tekst bestand hernoemen
        with open(os.path.join(output_dir, f"{base}_Llama2-13b.txt"), "a") as f:
            f.write("\nWilcoxon Signed-Rank Test Results:\n")
            for name1, data1, name2, data2 in pairs:
                stat, p = wilcoxon(data1, data2)
                f.write(f"{name1} vs {name2}: statistic={stat:.4f}, p-value={p:.4e}\n")

        raw_df = pd.DataFrame({
            'uuoo': uuoo,
            'uoou': uoou,
            'oouu': oouu,
            'ouuo': ouuo
        })
        raw_df.to_csv(os.path.join(raw_output_dir, f"{base}_raw_perplexities.csv"), index=False)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.boxplot([uuoo, uoou, oouu, ouuo], labels=['uuoo', 'uoou', 'oouu', 'ouuo'], showmeans=True)
        plt.title(f'Perplexiteit distributie van {base}')
        plt.ylabel('Perplexiteit')
        plt.grid(True)
        plt.tight_layout()

        # plt.savefig(os.path.join(output_dir, f"{base}_boxplot.png"))
        # plt.close()
