import streamlit as st
from boneagegrowth import run_prediction

st.set_page_config(page_title="골연령 기반 키 예측 AI")
st.title("🦴 골연령 기반 키 예측")
st.markdown("""
AI가 X-ray 이미지와 현재 키 정보를 기반으로 골연령을 분석하고, 
Bayley-Pinneau 성장률 표를 바탕으로 최종 키를 예측합니다.
""")

# 사용자 입력
uploaded_file = st.file_uploader("왼손 X-ray 이미지를 업로드하세요", type=["jpg", "png"])
sex = st.selectbox("성별을 선택하세요", ["male", "female"])
current_age = st.number_input("현재 나이 (예: 11.5)", min_value=0.0, max_value=20.0, step=0.1)
current_height = st.number_input("현재 키 (cm)", min_value=50.0, max_value=200.0, step=0.1)

# 예측 시작
if st.button("예측 시작"):
    if uploaded_file is None:
        st.warning("X-ray 이미지를 업로드해주세요.")
    else:
        with open("input_image.jpg", "wb") as f:
            f.write(uploaded_file.read())

        try:
            result = run_prediction(
                image_path="input_image.jpg",
                current_age=current_age,
                current_height=current_height,
                sex=sex
            )

            st.success("예측 완료!")
            st.write(f"🦴 AI 예측 골연령: **{result['bone_age']}세**")
            st.write(f"📈 성장군: **{result['group']}** (성장률 {result['rate']}%)")
            st.write(f"📏 예상 최종 키: **{result['final_height']} cm**")
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
