from utils import get_data
import pandas as pd

data = get_data()
print(data)

data.to_csv('file.csv', index=False)
