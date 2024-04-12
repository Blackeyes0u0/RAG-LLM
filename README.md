# bert q1,q2 비교

# 베이스 bert 


```bash
pip install -q sentence-transformers
```
```python
# 같은 piar 유사도 구하기 : v1평균 값 : 0.79386
# 같은 piar 유사도 구하기 : v2평균 값 : 0.78478
# 전체 샘플의 Cosine Similarity Score 평균 :  0.79386
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertConfig
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2

# Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기

model = SentenceTransformer('distiluse-base-multilingual-cased-v2') # 539M

model = SentenceTransformer('distiluse-base-multilingual-cased-v1') # 539M


# 문장 예시
preds = [
    "이번 경진대회는 질의 응답 처리를 수행하는 AI 모델을 개발해야합니다.",
    "데이콘은 플랫폼입니다."
]

gts = [
    "이번 경진대회의 주제는 도배 하자 질의 응답 AI 모델 개발입니다.",
    "데이콘은 국내 최대의 AI 경진대회 플랫폼입니다."
]

preds = list(train.질문_1)
gts  = list(train.질문_2)

# 샘플에 대한 Cosine Similarity 산식
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

sample_scores = []
for pred, gt in zip(preds, gts):
    # 생성된 답변 내용을 512 Embedding Vector로 변환
    pred_embed = model.encode(pred)
    gt_embed = model.encode(gt)
    
    print(gt_embed.shape)
    sample_score = cosine_similarity(gt_embed, pred_embed)
    # Cosine Similarity Score가 0보다 작으면 0으로 간주
    sample_score = max(sample_score, 0)
    print('예측 : ', pred)
    print('정답 : ', gt)
    print('Cosine Similarity Score : ', sample_score)
    print('-'*20)
    
    sample_scores.append(sample_score)

print(sample_scores)
print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))

```
![alt text](image-1.png)


#### 추가적으로 얼마나 서로 uniformity하게 임베딩이 구성되어 있는지 확인해 보았다. uniformity할 수록 다음 계산되는 값이 높길 바라는것이다.

```python
pred_embed = model.encode(preds)
gt_embed = model.encode(gts)

W = np.matmul(pred_embed,gt_embed.T)
print(W.shape)

score_i = []
score_j = []
for i in range(644):
    score_i.append(W[i][i]/sum(W[i][:]))

for i in range(644):
    score_j.append(W[i][i]/sum(W[:][i]))
    
print(sum(score_i)/644)
print(sum(score_j)/644)
plt.hist(score_i,bins=40)
```
![alt text](image-5.png)

평균 0.004 이므로 아쉬웠다. 물론 비슷한 쿼리가 존재를 많이 하겠지만, 그래도 이렇게 낮길 바라진않았다. 왜냐하면 1/664를 하면 0.0015이라 그렇게 높지 않다고 생각이 든다.

## version2도 살펴보았다.
![alt text](image-6.png)
![alt text](image-7.png)
어떻게 multilingual 모델에서 적절하게 uniformity, alignmnet를 구성하였을지 궁금했다.



---
---

# skt/kobert-v1

```bash
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

```python
# 단순 유사도 구하기 : base-encoder 값 : 0.7939
# skt/kobert  : 0.6921 - 0인 값들을 빼도 0.74 많이 부족함..

# 전체 샘플의 Cosine Similarity Score 평균 :  0.79386
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertConfig
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2

# Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
model = SentenceTransformer('distiluse-base-multilingual-cased-v1') # 539M


from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tokenizer.encode("한국어 모델을 공유합니다.")
#[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
model = BertModel.from_pretrained('skt/kobert-base-v1')


preds = list(train.질문_1)
gts  = list(train.질문_2)

# 샘플에 대한 Cosine Similarity 산식
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

sample_scores = []
for pred, gt in zip(preds, gts):
    # 생성된 답변 내용을 512 Embedding Vector로 변환
    
    inputs = tokenizer.batch_encode_plus([pred])
    out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))
    
    pred_embed = out.pooler_output.detach().numpy().reshape(-1)
    
    
    inputs = tokenizer.batch_encode_plus([gt])
    out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))
    gt_embed =  out.pooler_output.detach().numpy().reshape(-1)
    
    sample_score = cosine_similarity(gt_embed, pred_embed)
    # Cosine Similarity Score가 0보다 작으면 0으로 간주
    sample_score = max(sample_score, 0)
    print('예측 : ', pred)
    print('정답 : ', gt)
    print('Cosine Similarity Score : ', sample_score)
    print('-'*20)
    
    sample_scores.append(sample_score)
print(sample_scores)
print('전체 샘플의 Cosine Similarity Score 평균 : ', np.mean(sample_scores))
```

![alt text](image.png)




## skt/kobert uniformity

```python
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

def encode_texts(texts, max_length=512):
    all_encodings = torch.tensor([])
    for text in tqdm(texts):
        try:
            # Ensure consistent sequence length across all texts
            inputs = tokenizer.batch_encode_plus(
                [text], max_length=max_length, padding='max_length', truncation=True
            )

            out = BertModel.from_pretrained('skt/kobert-base-v1')(
                input_ids=torch.tensor(inputs['input_ids']),
                attention_mask=torch.tensor(inputs['attention_mask']),
            )

            # Store either pooler output or desired intermediate layer's output
            encoding = out.pooler_output.detach().numpy().reshape(-1)
            all_encodings = torch.concat((all_encodings,encoding),dim=0)
        except ValueError as e:
            print(f"Error encoding text: {text}")
            print(e)

    return all_encodings

# Example usage
encodings = encode_texts(preds)
print(encodings.shape)
encodings2 = encode_texts(gts)


W = pred_embed@gt_embed.T
print(W.shape)

score_i = []
score_j = []
for i in range(644):
    score_i.append(W[i][i]/sum(W[i][:]))

for i in range(644):
    score_j.append(W[i][i]/sum(W[:][i]))
    
print(sum(score_i)/644)
print(sum(score_j)/644)
plt.hist(score_i,bins=40)
```

## 유사도 비교 결과.

![alt text](image-3.png)

![alt text](image-4.png)

 multi lingual이 더 좋은 우수한 결과라고 생각된다.