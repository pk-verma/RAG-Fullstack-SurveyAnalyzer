# backend/app/data_processing.py
import pandas as pd

def load_data(file_path):
    # Load each sheet and process it into a list of documents
    dfs = pd.read_excel(file_path, sheet_name=None)
    data = []
    for sheet_name, df in dfs.items():
        for _, row in df.iterrows():
            row_text = " ".join(map(str, row.dropna().values))
            data.append({"sheet_name": sheet_name, "content": row_text})
    return data
