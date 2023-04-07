import pandas as pd

df = pd.read_parquet("data/full_cleaned.parquet", 
                     columns=["name", "is_full_name", "sex"])
