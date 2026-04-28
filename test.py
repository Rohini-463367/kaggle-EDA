from eda import load_data

csv_path, df = load_data()

print("Loaded file:", csv_path)
print(df.head())