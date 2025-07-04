import os
import re
import pandas as pd
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="sequential",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()

def clarify_requirement(req, max_new_tokens=125):
    prompt = (
        "Rewrite the following software requirement to make it clearer and more precise. The clarified sentence should be a complete and self-contained sentence. "
        "Do not add explanations, examples, or assumptions. Keep the meaning exactly the same.\n\n"
        f"{req}\n\nClarified:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clarified = output_text.split("Clarified:")[-1].strip().split("\n")[0]
    return clarified.strip()

def clean(x):
    if isinstance(x, str):
        return x.strip().replace('"', '')
    return x

requirements_dir = "requirements"
output_dir = "c_requirements"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(requirements_dir):
    if filename.endswith(".csv"):
        output_filename = re.sub(r'^\d+\s*-\s*', '', filename)
        input_path = os.path.join(requirements_dir, filename)
        output_path = os.path.join(output_dir, output_filename)

        try:
            df = pd.read_csv(input_path)
            df.columns = ["requirement"]

            clarified = []
            for req in df["requirement"]:
                clarified_sentence = clarify_requirement(req)
                cleaned = clarified_sentence.strip().strip('"')
                clarified.append(f'"{cleaned}"')

            df["control sentence"] = clarified
            df = df.rename(columns={"requirement": "underspecified sentence"})

            # df["underspecified sentence"] = df["underspecified sentence"].map(clean)
            # df["underspecified sentence"] = ['"' + str(x) + '"' for x in df["underspecified sentence"]]
            df["underspecified sentence"] = ['"' + str(x).strip().replace('"', '') + '"' for x in df["underspecified sentence"]]
            df["control sentence"] = df["control sentence"].map(clean)

            df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, quotechar='"')
        except Exception as e:
            print(":(")