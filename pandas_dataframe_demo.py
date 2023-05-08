import pandas as pd
import numpy as np

# DataFrame

# 利用numpy创建二维数据
arr = np.random.randn(20).reshape((4, 5))

index_name = ["r1", "r2", "r3", "r4"]
column_name = ["a", "b", "c", "d", "e"]

# 利用pandas创建DataFrame对象
df = pd.DataFrame(arr, index=index_name, columns=column_name)

print(df)
"""
print('-' * 50)

print(df.index)
print(df.columns)
print('-' * 50)

print(df.values)

print('-' * 50)

print(df.describe())

print('-' * 50)
"""
print(df.loc["r1"])
print(df.loc[:, "d"])
is_null_arr = df.isnull()
print(type(is_null_arr))
print(is_null_arr.iloc[0])

left = pd.DataFrame({
    "key": ["A0", "A1", "A2", "A3"],
    "B": ["B0", "B1", "B2", "B3"],
    "C": ["C0", "C1", "C2", "C3"],

    })

print(left)

left["B"] = "prefix." + left["B"] 

print('-' * 50)
print(left["B"])
print('-' * 50)

print(left)

print(df.head(2))

# 导出csv格式
#left.to_csv("left_data.csv")

# 导出json格式
#left.to_json("left.json")

