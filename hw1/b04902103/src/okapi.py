import numpy as np
from sklearn import preprocessing

def normalize(TF, IDF, k=1.5, b=0.75):
    docLen = TF.sum(1)
    avgLen = docLen.mean()
    
    TF = TF.tocoo()
    tmp1 = TF*(k+1)
    tmp2 = k*(1-b+b*docLen/avgLen)
    TF.data += np.array(tmp2[TF.row]).reshape(len(TF.data),)
    TF.data = tmp1.data/TF.data
    TF.data *= IDF[TF.col]
    TF = TF.tocsr()
    TF = preprocessing.normalize(TF, norm='l2', axis=1)
    return TF