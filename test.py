import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 28],
        'City': ['New York', 'London', 'Paris']}

df = pd.DataFrame(data)
filename = "output.csv"
df.to_csv(filename, index=False)

print(f"Data exported to {filename}")