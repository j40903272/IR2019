import os
import sys
import json

# textteaser
from textteaser import TextTeaser
tt = TextTeaser()
# gensim
from gensim.summarization.summarizer import summarize
# sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.summarizers.kl import KLSummarizer
LANGUAGE = "chinese"
SENTENCES_COUNT = 3
stemmer = Stemmer(LANGUAGE)
tokenizer = Tokenizer(LANGUAGE)
# bert-extractive-summarizer
from summarizer import SingleModel
model = SingleModel(model='bert-base-chinese', vector_size=768)
def overload(body, minl=10, maxl=600):
    return body.split('\n')
model.process_content_sentences = overload


import jieba
import re

def process_content(content):
    sent = content.replace('。', '。\n')
    sent = [i for i in re.split('，|;|,|\*|\n', content) if len(i) > 0]
    sent = list(jieba.cut("\n".join(sent)))
    sent = "".join(sent)
    return sent

def process_title(title):
    title = " ".join(title.split()[1:])
    titleword = list(jieba.cut(title))
    return " ".join(titleword)


def dada_summarize(content: str, title: str = "") -> dict :
    response = dict()
    content = process_content(content)
    title = process_title(title)
    
    # textrank [need newline to split sentence]
    response["textrank"] = summarize(content)
    
    # textteaser [need newline to split sentence]
    cnt = int(len(content.split('\n'))*0.3)
    response['textteaser'] = "\n".join(tt.summarize(title, content, count=cnt))
    
    ### sumy
    parser = PlaintextParser.from_string(content, tokenizer)
    
    # LSA
    summarizer = LsaSummarizer(stemmer)
    sentences = [str(i) for i in summarizer(parser.document, SENTENCES_COUNT)]
    response['lsa'] = "\n".join(sentences)
    
    # textrank2
    summarizer = TextRankSummarizer(stemmer)
    sentences = [str(i) for i in summarizer(parser.document, SENTENCES_COUNT)]
    response['textrank2'] = "\n".join(sentences)
    
    # lexrank
    summarizer = LexRankSummarizer(stemmer)
    sentences = [str(i) for i in summarizer(parser.document, SENTENCES_COUNT)]
    response['lexrank'] = "\n".join(sentences)
    
    # ruduction
    summarizer = ReductionSummarizer(stemmer)
    sentences = [str(i) for i in summarizer(parser.document, SENTENCES_COUNT)]
    response['reduction'] = "\n".join(sentences)
    
    #kl-sum
    summarizer = KLSummarizer(stemmer)
    sentences = [str(i) for i in summarizer(parser.document, SENTENCES_COUNT)]
    response['kl-sum'] = "\n".join(sentences)
    
    # bert
    response['bert'] = model(content, ratio=0.4)
    
    return response


def main():
    query = sys.argv[2]
    query_title = sys.argv[3] if len(sys.argv) > 3 else ""
    result = dada_summarize(query, query_title)
    with open("summary.json", 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()


dada_summarize("I like to eat popo.")
python dada_summarize.py "fuck u