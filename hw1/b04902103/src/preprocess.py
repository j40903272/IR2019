import os

def preprocess(modelDir):
    print("Preprocessing")
    
    vocab2idx = dict()
    idx2vocab = list()
    for e, line in enumerate(open(os.path.join(modelDir, 'vocab.all'), 'r')):
        idx2vocab.append(line.strip())
        vocab2idx[line.strip()] = e


    file2idx = dict()
    idx2file = list()
    for e, line in enumerate(open(os.path.join(modelDir, 'file-list'), 'r')):
        line = line.strip().split('/')[-1].lower()
        idx2file.append(line)
        file2idx[line] = e


    grams2idx = dict()
    idx2grams = list()
    f = open(os.path.join(modelDir, 'inverted-file'), 'r')
    for idx, line in enumerate(f):
        vocab1, vocab2, N = [int(i) for i in line.strip().split()]
        grams2idx[(vocab1, vocab2)] = idx
        idx2grams.append((vocab1, vocab2))
        for i in range(int(N)):
            f.readline()
        
    return vocab2idx, idx2file, grams2idx
