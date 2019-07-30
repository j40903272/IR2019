import os
import argparse
import numpy as np
import pandas as pd

import doc2vec, query2vec, okapi, predict
from preprocess import preprocess


parser = argparse.ArgumentParser(description='vsmodel')
parser.add_argument('-r', type=bool, default=True, dest="feedback")
parser.add_argument('-b', type=bool, default=True, dest="best")
parser.add_argument('-i', type=str, dest="queryFile")
parser.add_argument('-o', type=str, dest="rankList")
parser.add_argument('-m', type=str, dest="modelDir")
parser.add_argument('-d', type=str, dest="ntcirDir")
args = parser.parse_args()


vocab2idx, idx2file, grams2idx = preprocess(args.modelDir)
n_grams = len(grams2idx)
n_vocab = len(vocab2idx)
n_file = len(idx2file)


DOC_TF, DOC_IDF = doc2vec.parse(args.modelDir, n_file, n_grams)
Q_list, Q_TF, Q_IDF = query2vec.parse(args.queryFile, vocab2idx, grams2idx)

DVec = okapi.normalize(DOC_TF, DOC_IDF)
QVec = okapi.normalize(Q_TF, DOC_IDF)

pred = predict.rank(DVec, QVec, Q_list, idx2file)

with open(args.rankList, 'w') as f:
    print('results write to {}'.format(os.path.join(os.getcwd(), args.rankList)))
    print("query_id,retrieved_docs", file=f)
    for e, Q in enumerate(Q_list):
        print('{},{}'.format(Q.num, " ".join(pred[e])), file=f)



