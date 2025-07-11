import streamlit as st
import torch
import cv2
import numpy as np
from boneagegrowth import predict_final_height, crop_model, bone_model, device

st.title("🦴 AI 기반 골연령 기반 성장 예측 시스템")

# 사용자 입력
sex = st.selectbox("성별을 선택하세요", ["male", "female"])
current_age = st.number_input("현재 나이 (예: 11.5)", min_value=0.0, max_value=20.0, step=0.01)
current_height = st.number_input("현재 키 (cm)", min_value=50.0, max_value=200.0, step=0.1)
uploaded_file = st.file_uploader("왼손 X-ray 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로딩
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(img, caption="업로드된 X-ray 이미지", use_column_width=True)

    # 전처리 및 crop
    x = crop_model.preprocess(img)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    shape = torch.tensor([img.shape[:2]]).to(device)

    with torch.no_grad():
        coords = crop_model(x, shape)[0].cpu().numpy().astype(int)
    xmin, ymin, xmax, ymax = coords
    cropped = img[ymin:ymax, xmin:xmax]

    # 골연령 추정
    x = bone_model.preprocess(cropped)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    female_flag = torch.tensor([1 if sex == "female" else 0]).to(device)

    with torch.no_grad():
        pred = bone_model(x, female_flag)
    bone_age = round(pred.item() / 12, 2)

    # 최종 키 예측
    final_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

    # 결과 출력
    st.markdown("---")
    st.success("📈 예측 결과")
    st.write(f"**AI 예측 골연령**: {bone_age}세")
    st.write(f"**성장군**: {group} (성장률 {rate})")
    st.write(f"**예상 최종 키**: {final_height} cm")
