# !pip install transformers datasets accelerate

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import PatchTSTConfig, PatchTSTForPrediction
from datasets import load_dataset 

# 사전학습된 모델 다운로드
# Hugging Face Hub에는 다양한 데이터셋으로 사전학습된 PatchTST 모델들이 공개되어 있습니다.
# 본 실습에서는 ETTh1 데이터셋으로 학습된 모델을 사용합니다.

# Fine-tuned 모델 로드 (forecasting 태스크용)
model_name = "ibm/patchtst-etth1-forecasting"

model = PatchTSTForPrediction.from_pretrained(model_name)
print('✓ 모델 다운로드 및 로드 완료!')
print(f'모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}개')

# 모델 설정 확인
config = model.config 
print('=== 모델 설정 정보 ===')
print(f'입력 시퀀스 길이(context_length): {config.context_length}')
print(f'예측 길이(prediction_length): {config.prediction_length}')
print(f'패치 길이(patch_length): {config.patch_length}')
print(f'패치 간격(stride): {config.patch_stride}')
print(f'입력 채널 수(num_input_channels): {config.num_input_channels}')
print(f'히든 차원(d_model): {config.d_model}')
print(f'레이어 수(num_hidden_layers): {config.num_hidden_layers}')


# ETTh1 데이터셋 로드
dataset = load_dataset('ETDataset/ett', 'ETTh1')
print('✓ 데이터셋 다운로드 완료!')
print(f'Train 데이터: {len(dataset["train"])}개')
print(f'Test 데이터: {len(dataset["test"])}개')

# 테스트 데이터에서 첫 번째 샘플 추출 
test_sample = dataset['test'][0]

# 시계열 데이터 추출 (7개 채널)
ts_data = test_sample['target']
print(f'원본 시계열 길이: {len(ts_data)}')

# 데이터 형태 확인 및 시각화
ts_array = np.array(ts_data)
print(f'데이터 형태: {ts_array.shape}')  # [시퀀스 길이, 채널 수]

# ETTh1 데이터셋의 7개 채널 이름
channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# 7개 채널 시각화
fig, axes = plt.subplots(7, 1, figsize=(15, 12))
fig.suptitle('ETTh1 - 7 Channels', fontsize=16, y=0.995)

# 시각화할 데이터 포인트 수 (너무 많으면 일부만)
num_points = min(500, len(ts_array[0]))

for i in range(7):
    axes[i].plot(ts_array[i, :num_points], linewidth=1)
    axes[i].set_ylabel(channel_names[i], fontsize=10, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

    # 마지막 subplot에만 x축 라벨 표시
    if i == 6:
        axes[i].set_xlabel('Time Steps', fontsize=10)
    else:
        axes[i].set_xticklabels([])

plt.tight_layout()

plt.savefig('etth1_channels_visualization.png', dpi=150, bbox_inches='tight')
print('✓ 시각화 완료! (etth1_channels_visualization.png 저장됨)')
plt.show()

# 모델 입력을 위한 데이터 준비
# context_length만큼의 과거 데이터 사용
context_length = config.context_length
prediction_length = config.prediction_length

# 입력 데이터 준비 (선행하는 context_length 개 타임스텝)
# ts_data 형태: [7, total_timesteps]
past_values = np.array(ts_data)[:, -context_length-prediction_length:-prediction_length]  # [7, 512]
past_values = past_values.T  # [512, 7]로 전치
print(f'입력 데이터 형태 (정규화 전): {past_values.shape}')

# 데이터 정규화 (각 채널별로 평균=0, 표준편차=1로 스케일링)
mean = past_values.mean(axis=0, keepdims=True)  # [1, 7]
std = past_values.std(axis=0, keepdims=True)    # [1, 7]
past_values_normalized = (past_values - mean) / (std + 1e-8)  # epsilon으로 0으로 나누기 방지
print(f'정규화 완료 - Mean: {mean.flatten()[:3]}, Std: {std.flatten()[:3]}')

# 정답 데이터 준비 (입력 바로 다음의 prediction_length 개 타임스텝)
true_future = np.array(ts_data)[:, -prediction_length:]  # [7, prediction_length]
true_future = true_future.T  # [prediction_length, 7]로 전치
print(f'정답 데이터 형태: {true_future.shape}')



# PyTorch 텐서로 변환
# 형태: [batch_size, sequence_length, num_channels]
past_values_tensor = torch.tensor(past_values_normalized).unsqueeze(0).float()  # [1, 512, 7]
print(f'입력 텐서 형태: {past_values_tensor.shape}')

print(f'  - batch_size: {past_values_tensor.shape[0]}')
print(f'  - sequence_length: {past_values_tensor.shape[1]}')
print(f'  - num_channels: {past_values_tensor.shape[2]}')



# 모델을 evaluation 모드로 설정
model.eval() 

# 예측 수행 
with torch.no_grad():  # 그래디언트 계산 비활성화 (예측만 수행)     
    outputs = model(past_values=past_values_tensor)
    predictions = outputs.prediction_outputs
    
    print('✓ 예측 완료!')
    print(f'예측 결과 형태: {predictions.shape}')
    print(f'  - batch_size: {predictions.shape[0]}')
    print(f'  - prediction_length: {predictions.shape[1]}')
    print(f'  - num_channels: {predictions.shape[2]}')


# 예측값을 numpy 배열로 변환
predicted_values_normalized = predictions[0].cpu().numpy()

# 역정규화 (원래 스케일로 복원)
predicted_values = predicted_values_normalized * std + mean

# [prediction_length, num_channels]
print(f'변환된 예측값 형태: {predicted_values.shape}')

# 예측 성능 평가
mse = np.mean((predicted_values - true_future) ** 2)
mae = np.mean(np.abs(predicted_values - true_future))
rmse = np.sqrt(mse)

print(f'\n=== 예측 성능 ===')
print(f'MSE (Mean Squared Error): {mse:.4f}')
print(f'MAE (Mean Absolute Error): {mae:.4f}')
print(f'RMSE (Root Mean Squared Error): {rmse:.4f}')
print(f'\n첫 5개 타임스텝의 예측값 (채널 0):')
print(predicted_values[:5, 0])




# 모든 채널에 대한 서브플롯 생성
num_channels = config.num_input_channels
fig, axes = plt.subplots(num_channels, 1, figsize=(14, 3*num_channels))

for i in range(num_channels):
    ax = axes[i] if num_channels > 1 else axes

    # 과거, 예측, 정답 데이터
    past = past_values[:, i]
    pred = predicted_values[:, i]
    true = true_future[:, i]

    # 플롯
    ax.plot(range(len(past)), past, color='blue', linewidth=1.5, label='과거 (입력)')

    pred_start = len(past)
    ax.plot(range(pred_start, pred_start + len(pred)), pred, color='red', linewidth=1.5, linestyle='--', label='예측')
    ax.plot(range(pred_start, pred_start + len(true)), true, color='green', linewidth=1.5, linestyle='-', alpha=0.7, label='정답')

    ax.axvline(x=len(past), color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_ylabel(f'{channel_names[i]}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if i == 0:
        ax.legend(fontsize=9, loc='upper left')

    if i == num_channels - 1:
        ax.set_xlabel('Time Steps', fontsize=10)

plt.suptitle('PatchTST Forecasting - All Channels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('patchtst_forecast_result.png', dpi=150, bbox_inches='tight')
print('\n✓ 예측 결과 시각화 완료! (patchtst_forecast_result.png 저장됨)')
plt.show()