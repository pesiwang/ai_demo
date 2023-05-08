

def test_func(query:str, 
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n = 100):

    print(f"{top_n}")


test_func("test")


import pandas as pd

# create a DataFrame
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

print(df.head())

