import streamlit as st
import pandas as pd
import torch
import cv2
from transformers import AutoModel
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 성장률 표 데이터
male_bp_df = pd.DataFrame({
    "BA": [10 + 0.25*i for i in range(25)],
    "Retarded": [81.2, 81.6, 81.9, 82.3, 82.3, 82.7, 83.2, 83.9, 84.5, 85.2, 86.0, 86.6, 87.3, 88.0, 88.7, 89.3, 89.9, 90.5, 91.0, 91.5, 91.9, 92.3, 92.7, 93.0, 93.2],
    "Average":  [78.4, 79.1, 79.5, 80.0, 80.4, 81.2, 81.8, 82.7, 83.4, 84.3, 85.0, 85.7, 86.4, 87.1, 87.7, 88.3, 88.8, 89.3, 89.8, 90.2, 90.6, 91.0, 91.3, 91.5, 91.7],
    "Advanced": [74.7, 75.3, 75.8, 76.3, 76.7, 77.6, 78.6, 80.0, 80.9, 81.8, 82.7, 83.6, 84.4, 85.2, 85.9, 86.5, 87.1, 87.6, 88.1, 88.5, 88.9, 89.2, 89.4, 89.6, 89.8]
})

female_bp_df = pd.DataFrame({
    "BA": [10 + 0.25*i for i in range(21)],
    "Retarded": [87.6, 88.4, 88.6, 89.6, 91.8, 92.2, 92.9, 92.9, 93.2, 93.2, 93.3, 93.3, 93.3, 93.3, 93.2, 93.1, 93.0, 92.9, 92.7, 92.5, 92.3],
    "Average":  [86.2, 87.4, 88.4, 89.6, 90.6, 91.0, 91.4, 91.8, 92.2, 92.3, 92.5, 92.6, 92.6, 92.6, 92.5, 92.4, 92.3, 92.1, 91.9, 91.7, 91.5],
    "Advanced": [82.6, 84.1, 85.6, 87.0, 88.3, 88.7, 89.1, 89.7, 90.1, 90.4, 90.6, 90.8, 91.0, 91.1, 91.2, 91.3, 91.3, 91.3, 91.2, 91.1, 91.0]
})

# 키 예측 함수
def predict_final_height(current_age, current_height, bone_age, sex):
    df = male_bp_df if sex == "male" else female_bp_df

    # 골연령 클램핑
    bone_age = max(min(bone_age, df["BA"].max()), df["BA"].min())

    diff = bone_age - current_age
    if diff <= -1:
        group = "Retarded"
    elif diff >= 1:
        group = "Advanced"
    else:
        group = "Average"

    closest = df.iloc[(df["BA"] - bone_age).abs().argmin()]
    rate = closest[group]
    predicted = round(current_height * 100 / rate, 1)
    return predicted, group, rate

@st.cache_resource(show_spinner=False)
def load_models():
    crop_model = AutoModel.from_pretrained("ianpan/bone-age-crop", trust_remote_code=True).to(device)
    bone_model = AutoModel.from_pretrained("ianpan/bone-age", trust_remote_code=True).to(device)
    crop_model.eval()
    bone_model.eval()
    return crop_model, bone_model

def preprocess_image(image_file):
    img = Image.open(image_file).convert("L")  # 흑백으로 변환
    img = np.array(img)
    return img

def main():
    st.title("골연령 및 최종 키 예측 AI")

    crop_model, bone_model = load_models()

    sex = st.radio("성별을 선택하세요", ("male", "female"))
    current_age = st.number_input("현재 나이 (만 나이)", min_value=0.0, max_value=20.0, step=0.1)
    current_height = st.number_input("현재 키 (cm)", min_value=30.0, max_value=250.0, step=0.1)
    image_file = st.file_uploader("왼손 X-ray 이미지 업로드", type=["png", "jpg", "jpeg"])

    if st.button("예측 시작") and image_file is not None:
        img = preprocess_image(image_file)

        # crop_model preprocess
        x_crop = crop_model.preprocess(img)
        x_crop = torch.from_numpy(x_crop).unsqueeze(0).unsqueeze(0).float().to(device)
        img_shape = torch.tensor([img.shape[:2]]).to(device)

        with torch.no_grad():
            coords = crop_model(x_crop, img_shape)

        coords = coords[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = coords
        cropped_img = img[ymin:ymax, xmin:xmax]

        # bone_model preprocess
        x_bone = bone_model.preprocess(cropped_img)
        x_bone = torch.from_numpy(x_bone).unsqueeze(0).unsqueeze(0).float().to(device)

        female_flag = torch.tensor([1]).to(device) if sex == "female" else torch.tensor([0]).to(device)

        with torch.no_grad():
            pred = bone_model(x_bone, female_flag)

        bone_age = pred.item() / 12  # 개월 -> 연 단위

        predicted_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

        st.write(f"예측 골연령: {bone_age:.2f}세")
        st.write(f"성장군: {group}")
        st.write(f"성장률: {rate}")
        st.write(f"예상 최종 키: {predicted_height} cm")

if __name__ == "__main__":
    main()
