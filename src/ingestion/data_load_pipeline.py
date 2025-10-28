import pandas as pd
import json

def load_data(file):
    """Load fitness data from uploaded file (CSV/JSON)."""
    filename = file.name.lower()
    
    if filename.endswith(".csv"):
        return pd.read_csv(file)
    
    elif filename.endswith(".json"):
        data = json.load(file)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or JSON.")
