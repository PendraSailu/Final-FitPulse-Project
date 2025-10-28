import pandas as pd
import json

def load_json(path: str) -> pd.DataFrame:
    """Load fitness data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)