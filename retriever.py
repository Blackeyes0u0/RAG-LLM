from tokenizer.jh_tokenizer import JH_tokenizer
from tokenizer.jh_tokenizer import BM25_tokenizer
from rank_bm25 import BM25Okapi

def JH_retriever(docs,query,n:int=10,tokenizer_type='jh'):
    if tokenizer_type=='jh':
        # print('######### JH tokenizer ON  ##########')
        # tokenized_corpus = [JH_tokenizer(doc) for doc in docs]
        tokenized_corpus = JH_tokenizer(docs)
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = JH_tokenizer(query)
    else:
        # print('########  BM25 tokenizer ON  #######')
        tokenized_corpus = [BM25_tokenizer(doc) for doc in docs]

        
        bm25 = BM25Okapi(tokenized_corpus)
        # bm25.doc_len # 파싱된 문서의 길이
        # bm25.doc_freqs # 문서에 있는 각각의 토큰의 빈도 # 문서가 짧아서 의미가 없음.. 어떻게 해야할까?
        # bm25.idf # 토큰의 inverse term frequency를 계산해둠
        tokenized_query = BM25_tokenizer(query)
        
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_documents = bm25.get_top_n(tokenized_query, docs, n)
    return top_n_documents,doc_scores #list


def spliter(query,seperator):
    q = query
    maxn = len(seperator)-1
    for n,sep in enumerate(seperator):
        if type(q)==str:
            try:
                query_split = q.split(sep=sep)
                for i,q in enumerate(query_split):
                    if len(q)<2:
                        # del query_split[q]
                        query_split.remove(q)
                if len(query_split)>1:
                    return query_split
                else:
                    q = query_split[0]
                    if n ==maxn:
                        return q
            except:
                print('nveve')
        else:
            print('no')
            return q
            
# if __name__ =='__main__':
#     import pandas as pd
#     train = pd.read_csv('open/train.csv')
#     test = pd.read_csv('open/test.csv')
    
#     documents = list(train.질문_1) 
#     querys    = list(test.질문)

#     i = 0   
#     # query = querys[i]
#     # query = querys[i].split('?')[0]
    
#     seperator = ['?','그리고','.','또한',',']
#     query = spliter(querys[i],seperator)
#     print(query)
    
#     answer =''
#     if type(query)==list:
#         for q in query:
#             top_n_documents,doc_scores = JH_retriever(documents,q,n=1,tokenizer_type='jh')
#             answer+=' '+top_n_documents[0] # 첫번째를 합친다는 뜻
#         print(answer)
#     else:
#         top_n_documents,doc_scores = JH_retriever(documents,query,n=1,tokenizer_type='jh')
#         # for doc, score in zip(top_n_documents, doc_scores):
#         #     print(f"문서: {doc}")
#         #     print(f"점수: {score}")
#         #     print()
#         print(top_n_documents)