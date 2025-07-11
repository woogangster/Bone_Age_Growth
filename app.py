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

    if sex == "male":
        df = pd.DataFrame({"BA": ba_male, "Retarded": retarded_male, "Average": average_male, "Advanced": advanced_male})
    else:
        df = pd.DataFrame({"BA": ba_female, "Retarded": retarded_female, "Average": average_female, "Advanced": advanced_female})

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

# Streamlit Cloud에서 GPU 없음 → 강제로 CPU 설정
device = "cpu"
crop_model = AutoModel.from_pretrained("ianpan/bone-age-crop", trust_remote_code=True)
bone_model = AutoModel.from_pretrained("ianpan/bone-age", trust_remote_code=True)
crop_model.eval()
bone_model.eval()

def run_prediction(image_path, current_age, current_height, sex):
    import warnings
    warnings.filterwarnings("ignore")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = crop_model.preprocess(img)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    shape = torch.tensor([img.shape[:2]])

    with torch.no_grad():
        coords = crop_model(x, shape)[0].numpy().astype(int)
    xmin, ymin, xmax, ymax = coords
    cropped = img[ymin:ymax, xmin:xmax]

    x = bone_model.preprocess(cropped)
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    female_flag = torch.tensor([1 if sex == "female" else 0])

    with torch.no_grad():
        pred = bone_model(x, female_flag)
    bone_age = round(pred.item() / 12, 2)

    final_height, group, rate = predict_final_height(current_age, current_height, bone_age, sex)

    return {
        "bone_age": bone_age,
        "group": group,
        "rate": rate,
        "final_height": final_height
    }
