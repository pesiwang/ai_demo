import pandas as pd
import numpy as np

# Series

# 通过随机数创建numpy数组
arr = np.random.randn(6)
s1 = pd.Series(arr, index=["one", "two", "three", "four", "five", "six"])
print(s1)
print(s1[0])
print(s1["two"])
print("-" * 20)

# 通过arange()创建numpy数组
arr2 = np.arange(6)
s2 = pd.Series(arr2)
print(s2)
print("-" * 20)

# 通过ones创建numpy的全1数组
arr3 = np.ones(6)
s3 = pd.Series(arr3)
print(s3)
print("-" * 20)

# 通过zeros创建numpy的全零数组
arr4 = np.zeros(6)
s4 = pd.Series(arr4)
print(s4)

