import pandas as pd


def load():
    """
    df_norm is X =>training set
    """
    df = pd.read_table('data/data-numeric.txt', header=None, delim_whitespace=True)
    Y = df.iloc[:,-1]
    df = df.iloc[ :, :-1]
    df_norm = (df - df.min()) / (df.max() - df.min())
    return (df_norm, Y)