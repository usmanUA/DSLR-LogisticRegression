import numpy as np
import argparse
from DSLR.describe import describe
from DSLR.histogram import histogram
from DSLR.scatterplot import scatterplot
from DSLR.utils import loadDataset, getDataFeatures, getData

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



if __name__ == '__main__':
    main()
