import pandas as pd
from sklearn.datasets import load_svmlight_file

data, target = load_svmlight_file("/home/liangqian/文档/公开数据集/a9a/test.libsvm")
data = pd.DataFrame(data.toarray())
print(data)
