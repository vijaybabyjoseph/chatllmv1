# utils.py
import pandas as pd
import re

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]

def load_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
