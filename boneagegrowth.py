import streamlit as st
import pandas as pd
import torch
import cv2
from transformers import AutoModel
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 성장률 표 (GP-BP법 기준, 예시 일부 값)
female_bp_df = pd.DataFrame({
    "BA": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
    "Retarded": [82.5, 83.9, 85.3, 86.7, 88.1, 89.4, 90.8, 92.2, 93.6, 95.0, 96.3, 97.7, 99.1, 100.5, 101.9, 103.3, 104.7],
    "Average":  [84.2, 85.5, 86.9, 88.3, 89.7, 91.1, 92.5, 93.9, 95.2, 96.6, 98.0, 99.4, 100.8, 102.2, 103.6, 105.0, 106.4],
    "Advanced": [85.8, 87.2, 88.6, 90.0, 91.4, 92.8, 94.2, 95.5, 96.9, 98.3, 99.7, 101.1, 102.5, 103.9, 105.3, 106.7, 108.1]
})

male_bp_df = pd.DataFrame({
    "BA": [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0],
    "Retarded": [83.5, 84.9, 86.3, 87.7, 89.1, 90.5, 91.9, 93.3, 94.7, 96.1, 97.5, 98.9, 100.3, 101.7, 103.1, 104.5, 105.9],
    "Average":  [85.2, 86.6, 88.0, 89.4, 90.8, 92.2, 93.6, 95.0, 96.4, 97.8, 99.2, 100.6, 102.0, 103.4, 104.8, 106.2, 107.6],
    "Advanced": [86.8, 88.2, 89.6, 91.0, 92.4, 93.8, 95.2, 96.6, 98.0, 99.4, 100.8, 102.2, 103.6, 105.0, 106.4, 107.8, 109.2]
})

def predict_final_height(current_age, current_height, bone_age, sex):
    df = male_bp_df if sex == "male" else female_bp_df

    # 골연령 클램핑 (표 범위 밖일 때)
    bone_age = max(min(bone_age, df["BA"].max()), df["BA"].min())

    diff = bone_age - current_age
    if diff < -1:
        group = "Retarded"
    elif diff > 1:
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
    img = Image.open(image_file).convert("L")
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

        x_crop = crop_model.preprocess(img)
        x_crop = torch.from_numpy(x_crop).unsqueeze(0).unsqueeze(0).float().to(device)
        img_shape = torch.tensor([img.shape[:2]]).to(device)

        with torch.no_grad():
            coords = crop_model(x_crop, img_shape)

        coords = coords[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = coords
        cropped_img = img[ymin:ymax, xmin:xmax]

        x_bone = bone_model.preprocess(cropped_img)
        x_bone = torch.from_numpy(x_bone).unsqueeze(0).unsqueeze(0).float().to(device)

        female_flag = torch.tensor([1]).to(device) if sex == "female" else torch.tensor([0]).to(device)

        with torch.no_grad():
            pred = bone_model(x_bone, female_flag)

        bone_age = pred.item() / 12

        predicted_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

        st.write(f"예측 골연령: {bone_age:.2f}세")
        st.write(f"성장군: {group}")
        st.write(f"성장률: {rate}")
        st.write(f"예상 최종 키: {predicted_height} cm")

if __name__ == "__main__":
    main()
