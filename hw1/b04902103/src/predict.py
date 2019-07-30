import numpy as np

def rank(DVec, QVec, Q_list, idx2file):
    print("Predicting")
    
    nonzero = np.unique(QVec.indices)
    DVec = DVec[:,nonzero]
    QVec = QVec[:,nonzero]
    cosine = DVec*(QVec.transpose())

    prediction = []
    for e, Q in enumerate(Q_list):
        simList = []
        for fileIdx in range(len(idx2file)):
            simList.append((fileIdx, cosine[fileIdx, e]))
        simList.sort(key = lambda x: x[1], reverse = True)
        rank = [idx2file[i[0]] for i in simList]
        prediction.append(rank)
        
    return prediction