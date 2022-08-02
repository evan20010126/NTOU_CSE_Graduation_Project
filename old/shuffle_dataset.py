import pandas as pd


def shuffle(df):
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    print(df_shuffled)
