folder_path = "C:/Users/alvin_xtd9mn0/Documents/scriptie/type"

import pandas as pd
import os
import csv
import math

for filename in ["tcs.csv"]:
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL)

        # count the amount of types per file
        type_counts = df['type'].value_counts()
        print(f"File: {filename}")
        for t, count in type_counts.items():
            print(f"  Type: {t}, Count: {count}")
        print("\n")
        