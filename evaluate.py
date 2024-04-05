import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def args_parse():
    """ Parse command line params."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prediction_csv',
        type=Path,
        required=True,
        help='Path to prediction csv.'
    )

    return parser.parse_args()


def main():
    """ Main function. """
    args = args_parse()

    csv_file = pd.read_csv(args.prediction_csv, names=['ID', 'GT_CLASS', 'PREDICTION_CLASS'])
    gt = [x for x in csv_file['GT_CLASS']]
    pred = [x for x in csv_file['PREDICTION_CLASS']]

    accuracy = accuracy_score(gt, pred)
    print(f'Accuracy: {100 * accuracy} %')

    cm = confusion_matrix(gt, pred, labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    disp.plot(cmap='CMRmap')
    plt.show()


if __name__ == '__main__':
    main()