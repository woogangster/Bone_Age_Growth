import streamlit as st
import torch
import numpy as np
import cv2
from transformers import AutoModel
import pandas as pd

# 1. 성장률 BP표 정의
# (male_bp_df, female_bp_df 정의 — 위에서 쓰던 표 그대로 붙여넣기)

# 2. 최종 키 예측 함수
import pandas as pd

# GP-BP 표 데이터: 여성
female_bp_data = {
    "BA": [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5],
    "Retarded": [82.5, 83.9, 85.3, 86.7, 88.1, 89.4, 90.8, 92.2, 93.6, 95.0, 96.3, 97.7, 99.1, 100.5, 101.9, 103.3],
    "Average":   [84.2, 85.5, 86.9, 88.3, 89.7, 91.1, 92.5, 93.9, 95.2, 96.6, 98.0, 99.4, 100.8, 102.2, 103.6, 105.0],
    "Advanced":  [85.8, 87.2, 88.6, 90.0, 91.4, 92.8, 94.2, 95.5, 96.9, 98.3, 99.7, 101.1, 102.5, 103.9, 105.3, 106.7]
}

# GP-BP 표 데이터: 남성
male_bp_data = {
    "BA": [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5],
    "Retarded": [83.5, 84.9, 86.3, 87.7, 89.1, 90.5, 91.9, 93.3, 94.7, 96.1, 97.5, 98.9, 100.3, 101.7, 103.1, 104.5],
    "Average":   [85.2, 86.6, 88.0, 89.4, 90.8, 92.2, 93.6, 95.0, 96.4, 97.8, 99.2, 100.6, 102.0, 103.4, 104.8, 106.2],
    "Advanced":  [86.8, 88.2, 89.6, 91.0, 92.4, 93.8, 95.2, 96.6, 98.0, 99.4, 100.8, 102.2, 103.6, 105.0, 106.4, 107.8]
}

# 데이터프레임 생성
female_bp_df = pd.DataFrame(female_bp_data)
male_bp_df = pd.DataFrame(male_bp_data)

def predict_final_height(current_age, current_height, bone_age, sex):
    diff = bone_age - current_age
    if diff <= -1:
        group = "Retarded"
    elif diff >= 1:
        group = "Advanced"
    else:
        group = "Average"

    df = male_bp_df if sex == "male" else female_bp_df
    closest = df.iloc[(df["BA"] - bone_age).abs().argmin()]
    rate = closest[group]
    predicted = round(current_height * 100 / rate, 1)
    return predicted, group, rate

# 3. 모델 로딩
@st.cache_resource
def load_models():
    crop_model = AutoModel.from_pretrained("ianpan/bone-age-crop", trust_remote_code=True)
    bone_model = AutoModel.from_pretrained("ianpan/bone-age", trust_remote_code=True)
    crop_model.eval(); bone_model.eval()
    return crop_model, bone_model

crop_model, bone_model = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"
crop_model.to(device); bone_model.to(device)

# 4. UI 구성
st.title("AI 기반 골연령 & 키 예측기")
st.write("엑스레이를 업로드하면 골연령을 예측하고, 성장률 표를 활용해 최종 키를 계산해줍니다.")

uploaded_file = st.file_uploader("왼손 엑스레이 이미지를 업로드하세요", type=["jpg", "png"])
sex = st.radio("성별을 선택하세요", ["male", "female"])
current_age = st.number_input("현재 나이 (세)", min_value=1.0, max_value=20.0, step=0.1)
current_height = st.number_input("현재 키 (cm)", min_value=50.0, max_value=210.0, step=0.1)

# 5. 예측 실행
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # 골연령 예측
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
    final_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

    # 결과 출력
    st.success(f"예측된 골연령: {bone_age}세")
    st.info(f"성장군: {group} (성장률: {rate})")
    st.success(f"예상 최종 키: {final_height} cm")
