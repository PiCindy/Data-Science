import extraction
import clustering
import preprocessing
import classification

import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification', dest='classification', action='store_true')
    parser.add_argument('--no-classification', dest='classification', action='store_false')
    parser.add_argument('--clustering', dest='clustering', action='store_true')
    parser.add_argument('--no-clustering', dest='clustering', action='store_false')
    parser.set_defaults(classification=True, clustering=True)
    parser.add_argument('-p', '--parameters', type=int, nargs='+', help='Choose nb of persons per category and nb of sentences per person')

    args = parser.parse_args()

    data = pd.read_csv('data/data.csv', sep=',')

    if args.parameters:
        extraction.extraction(args.parameters[0], args.parameters[1])
        preprocessing.main(data)

    processed_data = pd.read_csv('data/processed_data.csv', sep=',')
    if args.clustering:
        clustering.main(processed_data)
    if args.classification:
        classification.main(processed_data)
