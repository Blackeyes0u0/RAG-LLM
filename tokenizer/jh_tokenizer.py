from konlpy.tag import Okt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering  # For completeness
from rank_bm25 import BM25Okapi

def JH_tokenizer(corpus:list): 
    model_ckpt = 'beomi/llama-2-ko-7b'
    model_ckpt = 'heegyu/koalpaca-355m'
    model_ckpt = 'beomi/KoAlpaca-KoRWKV-1.5B'
    model_ckpt = 'EleutherAI/polyglot-ko-1.3b' # best 
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    okt = Okt()
    
    # POS tagger initialization
    if type(corpus)==list:
        results = []
        inputs = tokenizer(corpus,add_special_tokens=True) #padding=True,
        for k,doc in enumerate(corpus):
            result = []
            for i in range(len(inputs['input_ids'][k])):
                decoded_token = tokenizer.decode(inputs['input_ids'][k][i])
                try:
                    pos_tag = okt.pos(decoded_token)[0][1]  # Assuming single token
                    if pos_tag in ['Noun','Verb']:
                        result.append(decoded_token)
                except Exception as e:
                    print(e)
                    # print(doc)
                    # print(decoded_token)
            results.append(result)
        return results
    
    if type(corpus)==str:
        result = []
        inputs = tokenizer(corpus,add_special_tokens=True) #padding=True,
        for i in range(len(inputs['input_ids'])):
            decoded_token = tokenizer.decode(inputs['input_ids'][i])
            try:
                pos_tag = okt.pos(decoded_token)[0][1]  # Assuming single token
                if pos_tag in ['Noun','Verb']:
                    result.append(decoded_token)
            except Exception as e:
                print(e)
                # print(corpus)
                # print(decoded_token)
        return result

def BM25_tokenizer(sent):
    okt = Okt()
    return [word for word, pos in okt.pos(sent) if pos in ['Noun','Verb']]# not in ["Josa", "Punctuation", "Eomi"]] #in ['Noun','Verb']]

# if __name__=='__main__':
#     import pandas as pd
#     train = pd.read_csv('../open/train.csv')
#     test = pd.read_csv('../open/test.csv')
    
#     documents = list(train.질문_1) 
#     querys    = list(test.질문)

#     i = 0   
#     query = querys[i]
#     print(query)

#     breakpoint()
#     tokenized_corpus = [BM25_tokenizer(doc) for doc in documents]