# using pymc to retrieve the data
import pymc as pm
import pandas as pd


data = pd.read_csv(pm.get_data("anemones.csv"))
data.to_csv("data/anemones.csv",index=False)
print("=== saved csv data ===")