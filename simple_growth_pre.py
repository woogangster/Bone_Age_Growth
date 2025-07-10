import streamlit as st

st.title("AI 키 성장 예측기")

name = st.text_input("이름을 입력하세요")
age = st.number_input("현재 나이 (세)", min_value=0, max_value=20)
gender = st.radio("성별", ["남자", "여자"])
height = st.number_input("현재 키 (cm)", min_value=0)
father_height = st.number_input("아버지 키 (cm)", min_value=0)
mother_height = st.number_input("어머니 키 (cm)", min_value=0)

if st.button("예상 키 예측"):
    parent_avg = (father_height + mother_height) / 2
    if gender == "남자":
        predicted = parent_avg + 6.5
    else:
        predicted = parent_avg - 6.5
    st.success(f"{name}님의 예상 최종 키는 약 {predicted:.1f}cm입니다!")
