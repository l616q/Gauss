import numpy as np
import pandas as pd
from utils.reduce_data import reduce_data


df = reduce_data("/home/liangqian/文档/公开数据集/test/train.csv")
df = pd.read_csv("/home/liangqian/文档/公开数据集/test/train.csv")
print(df.dtypes)
print(max(df['merchant_id']))
print(min(df['merchant_id']))
print(max(df['user_info.age_range']))
print(min(df['user_info.age_range']))
print(max(df['merchant_id']))
print(min(df['merchant_id']))
print(df.columns)
print(np.finfo(np.float16).max)
print(np.finfo(np.float16).min)
