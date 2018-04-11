from helpers.load_dataset import load
from helpers.lof import LocalOutlierFactor

def main():
    df = load()
    lof = LocalOutlierFactor(df=df, k=3, num_outliers=5)

if __name__ == '__main__':
    main()