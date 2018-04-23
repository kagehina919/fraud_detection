from helpers.load_dataset import load
from helpers.lof import LocalOutlierFactor


def compute_metrics(lof, Y):
    """

    :param lof:
    :param Y:
    :return:
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx, row in Y.iteritems():
        if idx in lof:
            if row == 2:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if row == 2:
                fn = fn + 1
            else:
                tn = tn + 1
    total = tp + fp + tn + fn
    precision = (float(tp + tn)/total)
    recall = (float(tp)/(tp + fn))
    f1_score = 2*precision*recall/float(precision+recall)
    print("Precision %s" % precision, "Recall %s" % recall, "F1-Score %s" % f1_score)

def main():
    df, Y = load()
    lof = LocalOutlierFactor(df=df)
    parameters = [
                (50, 300),
                (100, 300),
                (200, 300),
                (250, 300),
                (300, 300),
                (500, 300),
                (50,  150),
                (100, 150),
                (200, 150),
                (250, 150),
                (300, 150),
                (500, 150),]
    for idx, (k, num_outliers) in enumerate(parameters):
        print("Experiment:", idx, ", k =", k, ", num_outliers =", num_outliers)
        outliers =lof.find_outliers(k=k, num_outliers=num_outliers)
        compute_metrics(outliers, Y)
        print()


if __name__ == '__main__':
    main()