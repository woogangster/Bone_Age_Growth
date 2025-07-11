import streamlit as st
import pandas as pd
import torch
from transformers import AutoModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# 남아 BP 성장률 표
male_bp_df = pd.DataFrame({
    "BA": [
        6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50,
        8.75, 9.00, 9.25, 9.50, 9.75, 10.00, 10.25, 10.50, 10.75, 11.00,
        11.25, 11.50, 11.75, 12.00, 12.25
    ],
    "Retarded": [
        68.00, 69.00, 70.00, 70.90, 71.80, 72.80, 73.80, 74.70, 75.60, 76.50, 77.30,
        77.90, 78.60, 79.40, 80.00, 80.70, 81.20, 81.60, 81.90, 82.10, 82.30,
        82.70, 83.20, 83.90, 84.50, 85.20
    ],
    "Average": [
        69.50, 70.20, 70.90, 71.60, 72.30, 73.10, 73.90, 74.60, 75.20, 76.10, 76.90,
        77.70, 78.40, 79.10, 79.50, 80.00, 80.40, 81.20, 81.80, 82.70, 83.40,
        84.30, 84.80, 85.20, 83.40, 84.30
    ],
    "Advanced": [
        67.00, 67.60, 68.30, 68.90, 69.60, 70.30, 70.90, 71.50, 72.00, 72.80, 73.40,
        74.10, 74.70, 75.30, 75.80, 76.30, 76.70, 77.60, 78.60, 80.00, 80.90,
        81.80, 82.60, 83.50, 84.30, 85.20
    ],
})

# 여아 BP 성장률 표
female_bp_df = pd.DataFrame({
    "BA": [
        6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50,
        8.75, 9.00, 9.25, 9.50, 9.75, 10.00, 10.25, 10.50, 10.75, 11.00,
        11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00, 13.25, 13.50,
        13.75, 14.00, 14.25, 14.50, 14.75, 15.00, 15.25, 15.50, 15.75, 16.00,
        16.25, 16.50, 16.75, 17.00, 17.25, 17.50, 17.75, 18.00, 18.25, 18.50
    ],
    "Retarded": [
        73.30, 74.20, 75.10, 76.30, 77.00, 77.90, 78.80, 79.70, 80.40, 81.30, 82.30,
        83.60, 84.10, 85.10, 85.80, 86.60, 87.40, 88.40, 89.60, 90.70, 91.80,
        92.20, 92.60, 92.90, 93.20, 94.20, 94.90, 95.70, 96.40, 97.10, 97.70,
        98.10, 98.30, 98.60, 98.90, 99.20, 99.40, 99.50, 99.60, 99.70, 99.80,
        99.90, 99.90, 99.95, 100.00, 99.90, 99.95, 99.95, 100.00, 100.00, None
    ],
    "Average": [
        72.00, 72.90, 73.80, 75.10, 75.70, 76.50, 77.20, 78.20, 79.00, 80.10, 81.00,
        82.10, 82.70, 83.60, 84.40, 85.30, 86.20, 87.40, 88.40, 89.60, 90.60,
        91.00, 91.40, 91.80, 92.20, 93.20, 94.10, 95.00, 95.80, 96.70, 97.40,
        97.80, 98.00, 98.30, 98.60, 98.80, 99.00, 99.10, 99.30, 99.40, 99.50,
        99.60, 99.70, 99.80, 99.90, 99.90, 99.95, 99.95, 100.00, 100.00, None
    ],
    "Advanced": [
        None, None, 71.20, None, 71.20, 72.20, 73.20, 74.20, 75.00, 76.00, 77.10,
        78.40, 79.00, 80.00, 80.90, 81.90, 82.80, 84.10, 85.60, 87.00, 88.30,
        88.70, 89.10, 89.70, 90.10, 91.30, 92.40, 93.50, 94.50, 95.50, 96.30,
        96.80, 97.20, 97.70, 98.00, 98.30, 98.60, 98.80, 98.80, 99.20, 99.30,
        99.40, 99.50, 99.70, 99.80, 99.80, 99.95, 99.95, 100.00, 100.00, None
    ],
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
    img = Image.open(image_file).convert("L")  # 흑백 변환
    img = np.array(img)
    return img

def plot_growth(current_age, current_height, bone_age, predicted_height):
    ages = [current_age, bone_age, bone_age + 1]
    heights = [current_height, current_height * 100 / 92, predicted_height]  # 92는 예시 성장률

    fig, ax = plt.subplots()
    ax.plot(ages, heights, marker='o')
    ax.set_xlabel("나이 (세)")
    ax.set_ylabel("키 (cm)")
    ax.set_title("성장 예측 그래프")
    ax.grid(True)
    st.pyplot(fig)

def recommend_exercise(sex, age, growth_group):
    if growth_group == "Advanced":
        return "유산소 운동과 스트레칭을 꾸준히 해주세요."
    elif growth_group == "Average":
        return "근력 운동과 균형 잡힌 스트레칭을 권장합니다."
    else:  # Retarded
        return "가벼운 걷기와 충분한 휴식을 취하세요."

def recommend_diet(sex, age, growth_group):
    if growth_group == "Advanced":
        return "단백질과 칼슘이 풍부한 식사를 꾸준히 하세요."
    elif growth_group == "Average":
        return "균형 잡힌 식단과 충분한 비타민 섭취를 권장합니다."
    else:  # Retarded
        return "소화에 부담 없는 식사를 하면서 영양을 보충하세요."

def main():
    st.title("골연령 및 최종 키 예측 AI")

    crop_model, bone_model = load_models()

    sex = st.radio("성별을 선택하세요", ("male", "female"))
    current_age = st.number_input("현재 나이 (만 나이)", min_value=0.0, max_value=20.0, step=0.01)
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

        bone_age = pred.item() / 12  # 개월 -> 연 단위

        predicted_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

        st.write(f"예측 골연령: {bone_age:.2f}세")
        st.write(f"성장군: {group}")
        st.write(f"성장률: {rate}")
        st.write(f"예상 최종 키: {predicted_height} cm")

        plot_growth(current_age, current_height, bone_age, predicted_height)

        st.write("### 맞춤 운동 추천")
        st.write(recommend_exercise(sex, current_age, group))

        st.write("### 맞춤 식단 추천")
        st.write(recommend_diet(sex, current_age, group))

if __name__ == "__main__":
    main()
