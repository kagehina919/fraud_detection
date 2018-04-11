import pandas as pd


def load():
    """
    """
    df = pd.read_table('data/data-numeric.txt', header=None, delim_whitespace=True)
    df_norm = (df - df.min()) / (df.max() - df.min())
    # print(df_norm)
    return df_norm