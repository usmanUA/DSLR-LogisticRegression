import numpy as np
import argparse
from DSLR.describe import describe
from DSLR.histogram import histogram
from DSLR.pairplot import pairplot
from DSLR.scatterplot import scatterplot
from DSLR.utils import loadDataset, getDataFeatures, getData
from DSLR.train import trainModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help='input dataset')
    parser.add_argument("action", type=str, help='action identifier')
    args = parser.parse_args()
    dataset = loadDataset(args.dataset)
    features = getDataFeatures(dataset)
    data = getData(dataset)
    if args.action == 'describe':
        describe(features, data)
    elif args.action == 'histogram':
        histogram(features, data, "Marks", "Number of Students")
    elif args.action == 'scatterplot':
        scatterplot(features, data)
    elif args.action == 'pairplot':
        pairplot(features, data)
    elif args.action == 'train':
        trainModel(dataset[1:, 1:], features)
    else:
        print("Give a valid name for dataset and action")



if __name__ == '__main__':
    main()
