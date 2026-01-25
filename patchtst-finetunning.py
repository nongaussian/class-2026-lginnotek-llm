#!pip install transformers datasets==2.14.0 accelerate fsspec==2023.9.2
#!pip install yfinance

# GPU 사용 가능 여부 확인
import torch
if torch.cuda.is_available():
    print(f'✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠ GPU를 사용할 수 없습니다. CPU로 진행합니다.')
    print('런타임 > 런타임 유형 변경 > GPU를 선택하세요.')

import torch
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from transformers import (PatchTSTConfig, PatchTSTForPrediction, Trainer, TrainingArguments, EarlyStoppingCallback)
from torch.utils.data import Dataset


# 주식 종목 설정
ticker = 'AAPL'

# 데이터 기간 설정 (최근 2년)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
print(f'주식 데이터 다운로드 중: {ticker}')
print(f'기간: {start_date.date()} ~ {end_date.date()}')

# 데이터 다운로드
stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print(f'\n✓ 데이터 다운로드 완료!')
print(f'총 데이터 포인트: {len(stock_data)}개')

# 샘플 출력
print(stock_data.head())

# 사용할 특징 선택 (OHLCV - Open, High, Low, Close, Volume)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
df = stock_data[features].copy()
# 결측치 제거
df = df.dropna()
print('=== 데이터 정보 ===')
print(f'사용 특징: {features}')
print(f'특징 개수: {len(features)}개')
print(f'데이터 크기: {df.shape}')
print(f'\n기초 통계:')
print(df.describe())

# 데이터 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)
print('✓ 데이터 정규화 완료')
print(f'정규화 후 데이터 형태: {scaled_data.shape}')
print(f'정규화 후 평균: {scaled_data.mean(axis=0)}')
print(f'정규화 후 표준편차: {scaled_data.std(axis=0)}')


# 데이터 분할 비율 설정
train_ratio = 0.7 # 70% 학습
val_ratio = 0.15  # 15% 검증
test_ratio = 0.15 # 15% 테스트
# 분할 인덱스 계산
total_len = len(scaled_data)
train_end = int(total_len * train_ratio)
val_end = int(total_len * (train_ratio + val_ratio))

# 데이터 분할
train_data = scaled_data[:train_end]
val_data = scaled_data[train_end:val_end]
test_data = scaled_data[val_end:]
print('=== 데이터 분할 결과 ===')
print(f'전체 데이터: {total_len}개')
print(f'학습 데이터: {len(train_data)}개 ({len(train_data)/total_len*100:.1f}%)')
print(f'검증 데이터: {len(val_data)}개 ({len(val_data)/total_len*100:.1f}%)')
print(f'테스트 데이터: {len(test_data)}개 ({len(test_data)/total_len*100:.1f}%)')


class TimeSeriesDataset(Dataset):
    """
    시계열 데이터를 위한 PyTorch Dataset
    Parameters:
    -----------
    data : numpy.ndarray         정규화된 시계열 데이터 [timesteps, features]
    context_length : int         입력으로 사용할 과거 데이터 길이
    prediction_length : int      예측할 미래 데이터 길이
    """
    
    def __init__(self, data, context_length, prediction_length):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        # 생성 가능한 샘플 수 계산
        self.total_length = context_length + prediction_length
        self.num_samples = len(data) - self.total_length + 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 시작 인덱스
        start_idx = idx
        # 과거 데이터 (입력)
        past_values = self.data[start_idx:start_idx + self.context_length]
        # 미래 데이터 (정답)
        future_values = self.data[
            start_idx + self.context_length:
            start_idx + self.total_length
        ]
        return {
            'past_values': torch.tensor(past_values, dtype=torch.float32),  # [time, features]
            'future_values': torch.tensor(future_values, dtype=torch.float32)  # [time, features]
        }
        
# 시퀀스 길이 설정
context_length = 128
# 과거 128 타임스텝 사용
prediction_length = 24
# 24 타임스텝 예측 (약 1개월)
print('=== 하이퍼파라미터 ===')
print(f'입력 길이 (context_length): {context_length}')
print(f'예측 길이 (prediction_length): {prediction_length}')
print(f'특징 개수 (num_input_channels): {len(features)}')


# 학습/검증/테스트 Dataset 생성
train_dataset = TimeSeriesDataset(
    train_data,
    context_length,
    prediction_length
) 
val_dataset = TimeSeriesDataset(
    val_data,
    context_length,
    prediction_length
)
test_dataset = TimeSeriesDataset(
    test_data,
    context_length,
    prediction_length
)
print('=== Dataset 생성 완료 ===')
print(f'학습 샘플 수: {len(train_dataset)}')
print(f'검증 샘플 수: {len(val_dataset)}')
print(f'테스트 샘플 수: {len(test_dataset)}')

# 샘플 데이터 확인
sample = train_dataset[0]
print(f'\n샘플 데이터 형태:')
print(f'  past_values: {sample["past_values"].shape}')      # [context_length, features]
print(f'  future_values: {sample["future_values"].shape}')  # [prediction_length, features]



# 사전학습 모델 이름
pretrained_model = "ibm-granite/granite-timeseries-patchtst"
print(f'사전학습 모델 로드 중: {pretrained_model}')

# 사전학습 모델의 config를 로드하고 우리 데이터에 맞게 수정
config = PatchTSTConfig.from_pretrained(pretrained_model)
config.num_input_channels = len(features)    # 5개 (OHLCV)
config.context_length = context_length       # 128
config.prediction_length = prediction_length # 24

# 수정된 config로 모델 생성 (사전학습 가중치 중 호환되는 부분만 로드)
model = PatchTSTForPrediction.from_pretrained(
    pretrained_model,
    config=config,
    ignore_mismatched_sizes=True  # 크기가 다른 레이어는 새로 초기화
)
print('✓ 사전학습 모델 로드 완료')
print(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}개')

print('=== 업데이트된 모델 설정 ===')
print(f'입력 채널 수: {model.config.num_input_channels}')
print(f'Context 길이: {model.config.context_length}')
print(f'예측 길이: {model.config.prediction_length}')
print(f'패치 길이: {model.config.patch_length}')
print(f'패치 간격: {model.config.patch_stride}')
print(f'히든 차원: {model.config.d_model}')
print(f'레이어 수: {model.config.num_hidden_layers}')

# GPU 사용 가능하면 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# 학습 하이퍼파라미터 설정
training_args = TrainingArguments(
    output_dir='./patchtst_stock_finetuned',    # 모델 저장 경로    

    # 학습 설정
    num_train_epochs=50,               # 에포크 수 (early stopping으로 조기 종료)
    per_device_train_batch_size=32,    # 배치 크기
    per_device_eval_batch_size=64,     # 평가 배치 크기

    # 학습률 및 옵티마이저
    learning_rate=1e-4,                # 학습률 (fine-tuning은 작게 설정)
    warmup_steps=50,                   # 워밍업 스텝
    weight_decay=0.01,                 # 가중치 감쇠 (과적합 방지)

    # 평가 및 저장
    eval_strategy='epoch',             # 에포크마다 평가
    save_strategy='epoch',             # 에포크마다 저장
    save_total_limit=3,                # 최근 3개 체크포인트만 유지
    
    # 로깅
    logging_dir='./logs',
    logging_strategy='epoch',
    logging_steps=10,
    
    # 성능 최적화
    fp16=torch.cuda.is_available(),    # GPU 있으면 mixed precision 사용
    dataloader_num_workers=2,          # 재현성     
    seed=42, 
)


# Early Stopping 콜백
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5,       # 5 에포크 동안 개선 없으면 중단
    early_stopping_threshold=0.0001  # 최소 개선 임계값
)


# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping],
) 

print('✓ Trainer 생성 완료')
print('\n' + '='*50)
print('Fine-tuning 학습 시작!')
print('='*50)  
# 학습 시작
train_result = trainer.train()
print('\n' + '='*50)
print('✓ Fine-tuning 학습 완료!')
print('='*50)

# 모델 저장
trainer.save_model('./patchtst_stock_final')
print('\n✓ Fine-tuned 모델 저장 완료: ./patchtst_stock_final')



# 테스트 데이터로 평가
test_results = trainer.evaluate(test_dataset)
print('=== 테스트 결과 ===')
print(f'테스트 손실 (MSE): {test_results["eval_loss"]:.6f}')



# 예측 모드로 설정
model.eval()
# 테스트 샘플 선택
test_sample_idx = 0
test_sample = test_dataset[test_sample_idx]
# 예측 수행
with torch.no_grad():
    past_values = test_sample['past_values'].unsqueeze(0).to(device)  # [1, features, time]
    outputs = model(past_values=past_values)
    predictions = outputs.prediction_outputs[0]  # [time, features]
    
    # CPU로 이동 및 numpy 변환
    predictions = predictions.cpu().numpy()
    true_values = test_sample['future_values'].numpy()  # [time, features]
    print('✓ 예측 완료')
    print(f'예측 결과 형태: {predictions.shape}')
    print(f'실제 값 형태: {true_values.shape}')


# 역정규화
def inverse_transform(data, scaler):
    return scaler.inverse_transform(data)

# 예측값과 실제값 역정규화
predictions_orig = inverse_transform(predictions, scaler)
true_values_orig = inverse_transform(true_values, scaler)
print('✓ 역정규화 완료')
print(f'\n원래 스케일의 종가(Close) 예측:')
print(f'첫 5개 예측값: {predictions_orig[:5, 3]}')
print(f'첫 5개 실제값: {true_values_orig[:5, 3]}')



# 종가 예측 시각화
close_idx = features.index('Close')  # Close 컬럼의 인덱스 (3)
plt.figure(figsize=(14, 6))

# 과거 데이터 (context)
past_close = test_sample['past_values'][close_idx].cpu().numpy()
past_close_orig = inverse_transform(scaler.mean_[close_idx] + past_close * scaler.scale_[close_idx], scaler )[close_idx]
time_steps_past = range(len(past_close))
time_steps_future = range(len(past_close), len(past_close) + prediction_length)

# 그래프 그리기
plt.plot(time_steps_past, inverse_transform(past_close.reshape(-1, 1), scaler), label='Past Close Price', color='blue', linewidth=2)
plt.plot(time_steps_future, true_values_orig[:, close_idx], label='Actual Future Close', color='green', linewidth=2)
plt.plot(time_steps_future, predictions_orig[:, close_idx], label='Predicted Close', color='red', linewidth=2, linestyle='--')

# 경계선
plt.axvline(x=len(past_close), color='gray', linestyle=':', linewidth=2, label='Prediction Start')
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel(f'{ticker} Close Price ($)', fontsize=12)
plt.title(f'{ticker} Stock Price Prediction (Fine-tuned PatchTST)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()




def predict_and_show(test_sample_idx):
  # 예측 모드로 설정
  model.eval()

  # 테스트 샘플 선택
  test_sample = test_dataset[test_sample_idx]

  # 예측 수행
  with torch.no_grad():
    past_values = test_sample['past_values'].unsqueeze(0).to(device)  # [1, features, time]
    outputs = model(past_values=past_values)
    predictions = outputs.prediction_outputs[0]  # [time, features]

    # CPU로 이동 및 numpy 변환
    predictions = predictions.cpu().numpy()
    true_values = test_sample['future_values'].numpy()  # [time, features]

  # 종가 예측 시각화
  close_idx = features.index('Close')  # Close 컬럼의 인덱스 (3)
  plt.figure(figsize=(14, 6))

  # 과거 데이터 (context)
  past_close = test_sample['past_values'][:, close_idx].cpu().numpy()
  #past_close_orig = inverse_transform(scaler.mean_ + past_close * scaler.scale_, scaler)[close_idx]
  time_steps_past = range(len(past_close))
  time_steps_future = range(len(past_close), len(past_close) + prediction_length)

  # 그래프 그리기
  plt.plot(time_steps_past, past_close, label='Past Close Price', color='blue', linewidth=2)
  plt.plot(time_steps_future, true_values[:, close_idx], label='Actual Future Close', color='green', linewidth=2)
  plt.plot(time_steps_future, predictions[:, close_idx], label='Predicted Close', color='red', linewidth=2, linestyle='--')

  # 경계선
  plt.axvline(x=len(past_close), color='gray', linestyle=':', linewidth=2, label='Prediction Start')
  plt.xlabel('Time Steps', fontsize=12)
  plt.ylabel(f'{ticker} Close Price (scaled)', fontsize=12)
  plt.title(f'{ticker} Stock Price Prediction (Fine-tuned PatchTST)', fontsize=14, fontweight='bold')
  plt.legend(fontsize=11)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()