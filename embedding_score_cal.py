# 단순 유사도 구하기 : 평균 값 : 0.79386
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
# model = 

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