import os
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

def parse(modelDir, n_file, n_grams):
    print("Calculating document tf-idf")
    
    col = []
    row = []
    data = []
    DOC_IDF = []
    
    f = open(os.path.join(modelDir, 'inverted-file'), 'r')
    for idx, line in enumerate(tqdm(f, total=n_grams)):
        vocab1, vocab2, N = [int(i) for i in line.strip().split()]
        for i in range(int(N)):
            fileId, cnt = f.readline().strip().split()
            fileId, cnt = int(fileId), int(cnt)
            data.append(cnt)
            row.append(fileId)
            col.append(idx)

        DOC_IDF.append(np.log( (n_file+1)/(N+1)) + 1)
        
    DOC_TF = csr_matrix((data, (row, col)), shape=(n_file, n_grams), dtype='float')
    
    return DOC_TF, np.array(DOC_IDF)
        