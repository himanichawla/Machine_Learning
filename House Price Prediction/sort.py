import pandas as pd
import numpy as np

## General Statistics

# Read into dataframe and check
from_csv = pd.read_csv('test.csv', encoding='utf-8',lineterminator ='\n')

sort_csv=from_csv.sort_values(by='SalePrice', ascending=1)

sort_csv.to_csv("test_sorted.csv")