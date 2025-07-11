import torch
import pandas as pd
import cv2
import numpy as np
from transformers import AutoModel

# 정확한 성장률 표 (여아용)
female_bp_df = pd.DataFrame({
    "BA": [
        6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50,
        9.75, 10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75, 12.00
    ],
    "Retarded": [
        73.3, 74.2, 75.1, 76.3, 77.0, 77.9, 78.8, 79.7, 80.4, 81.3, 82.3, 83.6, 84.1, 85.1, 85.8,
        86.6, 87.0, 88.4, 89.6, 90.7, 91.8, 92.2, 92.9, 92.9, 93.2
    ],
    "Average": [
        72.0, 72.9, 73.8, 75.1, 75.7, 76.5, 77.2, 78.0, 79.0, 80.0, 81.0, 82.1, 82.7, 83.6, 84.4,
        85.3, 86.2, 87.4, 88.4, 89.6, 90.6, 91.0, 91.4, 91.8, 92.2
    ],
    "Advanced": [
        70.0, 71.0, 72.0, 71.2, 71.2, 72.2, 73.2, 74.2, 75.0, 76.0, 77.0, 78.1, 79.0, 80.0, 80.9,
        81.9, 82.6, 84.1, 85.6, 87.0, 88.3, 88.7, 89.1, 89.7, 90.1
    ]
})

# 성장률 표 (남아용)
male_bp_df = pd.DataFrame({
    "BA": [10 + 0.25*i for i in range(25)],
    "Retarded": [81.2, 81.6, 81.9, 82.3, 82.3, 82.7, 83.2, 83.9, 84.5, 85.2, 86.0, 86.6, 87.3, 88.0, 88.7, 89.3, 89.9, 90.5, 91.0, 91.5, 91.9, 92.3, 92.7, 93.0, 93.2],
    "Average":  [78.4, 79.1, 79.5, 80.0, 80.4, 81.2, 81.8, 82.7, 83.4, 84.3, 85.0, 85.7, 86.4, 87.1, 87.7, 88.3, 88.8, 89.3, 89.8, 90.2, 90.6, 91.0, 91.3, 91.5, 91.7],
    "Advanced": [74.7, 75.3, 75.8, 76.3, 76.7, 77.6, 78.6, 80.0, 80.9, 81.8, 82.7, 83.6, 84.4, 85.2, 85.9, 86.5, 87.1, 87.6, 88.1, 88.5, 88.9, 89.2, 89.4, 89.6, 89.8]
})

# 키 예측 함수
def predict_final_height(current_age, current_height, bone_age, sex):
    df = male_bp_df if sex == "male" else female_bp_df

    # 골연령 범위 제한
    bone_age = max(min(bone_age, df["BA"].max()), df["BA"].min())

    # 성장군 판별
    diff = bone_age - current_age
    if diff >= 1:
        group = "Advanced"
    elif diff <= -1:
        group = "Retarded"
    else:
        group = "Average"

    # 가장 가까운 BA 값에서 성장률 추출
    idx = (df["BA"] - bone_age).abs().idxmin()
    rate = df.loc[idx, group]

    predicted = round(current_height * 100 / rate, 1)
    return predicted, group, rate

# 모델 불러오기
device = "cuda" if torch.cuda.is_available() else "cpu"
crop_model = AutoModel.from_pretrained("ianpan/bone-age-crop", trust_remote_code=True).to(device)
bone_model = AutoModel.from_pretrained("ianpan/bone-age", trust_remote_code=True).to(device)
crop_model.eval()
bone_model.eval()

# 사용자 입력
print("== 키 예측 프로그램 ==")
sex = input("성별을 입력하세요 (male/female): ").strip().lower()
current_age = float(input("현재 나이 (예: 11.5): "))
current_height = float(input("현재 키 (cm): "))
image_path = input("왼손 X-ray 이미지 경로 (예: /content/BAT1.jpg): ")

# 이미지 로딩 및 전처리
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
x = crop_model.preprocess(img)
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
shape = torch.tensor([img.shape[:2]]).to(device)

with torch.no_grad():
    coords = crop_model(x, shape)[0].cpu().numpy().astype(int)
xmin, ymin, xmax, ymax = coords
cropped = img[ymin:ymax, xmin:xmax]

x = bone_model.preprocess(cropped)
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
female_flag = torch.tensor([1 if sex == "female" else 0]).to(device)

with torch.no_grad():
    pred = bone_model(x, female_flag)
bone_age = round(pred.item() / 12, 2)

# 최종 키 예측
final_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

# 결과 출력
print("\n[예측 결과]")
print(f"현재 나이: {current_age}세")
print(f"현재 키: {current_height}cm")
print(f"AI 예측 골연령: {bone_age}세")
print(f"성장군: {group} (성장률 {rate})")
print(f"예상 최종 키: {final_height} cm")
