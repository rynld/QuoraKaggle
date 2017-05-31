import pandas as pd


df = pd.read_csv("./input/test_features.csv", encoding="ISO-8859-1")

print(df.sample())
print("--"*20)
print(df.head())
print("--"*20)
print(df.columns)

