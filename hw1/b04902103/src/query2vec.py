import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.sparse import csr_matrix


class query():
    def __init__(self, num, title, question, narrative, concepts, doc):
        self.num = num
        self.title = title
        self.question = question
        self.narrative = narrative
        self.concepts = concepts
        self.doc = doc



def cal_query_tfidf(Q, vocab2idx, grams2idx, Tweight = 5, Dweight = 3):
    
    title = Q.title
    doc = Q.doc
    tf = np.zeros(len(grams2idx))
    idf = np.zeros(len(grams2idx))
    
    # unigram
    for i in title:
        if i in vocab2idx:
            target = (vocab2idx[i], -1)
            if target in grams2idx:
                tf[grams2idx[target]] += Tweight
                idf[grams2idx[target]] = 1
    # bigram
    for i in range(len(title)-1):
        if (title[i] in vocab2idx) and (title[i+1] in vocab2idx):
            vid1 = vocab2idx[title[i]]
            vid2 = vocab2idx[title[i+1]]
            target = (vid1, vid2)
            if target in grams2idx:
                tf[grams2idx[target]] += Tweight
                idf[grams2idx[target]] = 1
    
    # unigram
    for i in doc:
        if i in vocab2idx:
            target = (vocab2idx[i], -1)
            if target in grams2idx:
                tf[grams2idx[target]] += Dweight
                idf[grams2idx[target]] = 1
    # bigram
    for i in range(len(doc)-1):
        if (doc[i] in vocab2idx) and (doc[i+1] in vocab2idx):
            vid1 = vocab2idx[doc[i]]
            vid2 = vocab2idx[doc[i+1]]
            target = (vid1, vid2)
            if target in grams2idx:
                tf[grams2idx[target]] += Dweight
                idf[grams2idx[target]] = 1
    
    return tf, idf
        
    
    

def parse(queryFile, vocab2idx, grams2idx):
    print("Calculating query tf-idf")
    
    f = open(queryFile, 'r', encoding='utf-8')
    xml=ET.parse(f).getroot()
    f.close()
    topics= xml.findall("topic")

    Q_TF = list()
    Q_IDF = list()
    Q_list = list()

    for topic in tqdm(topics):
        num = topic.find("number").text.strip().split("ZH")[1]
        title = topic.find("title").text.strip()
        question = topic.find("question").text.strip()
        narrative = topic.find("narrative").text.split("。")[0]
        concepts = topic.find("concepts").text.strip()
        doc = " ".join([question, narrative, concepts])
        doc = doc.replace("查詢","").replace("相關文件內容","").replace("應","").replace("包括","").replace("應說明","")

        Q = query(num, title, question, narrative, concepts, doc)
        Q_list.append(Q)

        tf, idf = cal_query_tfidf(Q, vocab2idx, grams2idx)
        Q_TF.append(tf)
        Q_IDF.append(idf)
    
    
    Q_IDF = np.stack(Q_IDF).sum(0)
    Q_IDF = np.log( (len(topics)+1) / (Q_IDF+1)) + 1
    Q_TF = np.stack(Q_TF)
    Q_TF = csr_matrix(Q_TF)
    
    return Q_list, Q_TF, Q_IDF
