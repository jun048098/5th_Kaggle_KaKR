# 5th_Kaggle_KaKR
![kakr_5](https://github.com/jun048098/5th_Kaggle_KaKR/assets/96534680/6dc20c9d-25dd-4b78-a1ca-5a3e7d6f87be)


[The 5th Kaggle Competition Challenge with KaKR](https://www.kaggle.com/competitions/the-5th-kaggle-competition-challenge-with-kakr/overview)

## Competition

|**대회 목표**| **`악성 댓글 유해도 예측`** |
| :---: | --- |
|**구현 내용**| Hugging Face의 Pretrained 모델을 Trainer로 Fine-tuning|
|**개발 환경**| `GPU` : 3070ti, `Kaggle notebook` : P100|

### How to Train
```
# train.py 실행
# yaml 변경 예시
python train.py --config_yaml base.yaml
```