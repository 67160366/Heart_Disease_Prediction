import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

model = load_model()

# --- Header ---
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("""
ระบบประเมินความเสี่ยงโรคหัวใจเบื้องต้นด้วย Machine Learning
กรอกข้อมูลด้านล่างเพื่อประเมินความเสี่ยง
""")

st.divider()

# --- Input Form ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ข้อมูลทั่วไป")
    age = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("เพศ", options=["M", "F"], format_func=lambda x: "ชาย" if x == "M" else "หญิง")
    chest_pain = st.selectbox("ประเภทอาการเจ็บหน้าอก",
                              options=["ATA", "NAP", "ASY", "TA"],
                              format_func=lambda x: {
                                  "TA": "TA — Typical Angina (เจ็บแบบ classic)",
                                  "ATA": "ATA — Atypical Angina (เจ็บแบบไม่ตรงตำรา)",
                                  "NAP": "NAP — Non-Anginal Pain (เจ็บไม่เกี่ยวกับหัวใจ)",
                                  "ASY": "ASY — Asymptomatic (ไม่มีอาการ)"
                              }[x])

with col2:
    st.subheader("ผลตรวจเลือด & ความดัน")
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", min_value=0, max_value=300, value=120, step=1)
    cholesterol = st.number_input("คอเลสเตอรอล (mg/dl)", min_value=0, max_value=700, value=200, step=1)
    fasting_bs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl?",
                              options=[0, 1],
                              format_func=lambda x: "ไม่ใช่" if x == 0 else "ใช่")

with col3:
    st.subheader("ผลตรวจหัวใจ")
    resting_ecg = st.selectbox("ผล ECG ขณะพัก",
                               options=["Normal", "ST", "LVH"],
                               format_func=lambda x: {
                                   "Normal": "Normal (ปกติ)",
                                   "ST": "ST-T wave abnormality",
                                   "LVH": "LVH (Left Ventricular Hypertrophy)"
                               }[x])
    max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (bpm)", min_value=50, max_value=250, value=150, step=1)
    exercise_angina = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย?",
                                   options=["N", "Y"],
                                   format_func=lambda x: "ไม่มี" if x == "N" else "มี")
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope",
                            options=["Up", "Flat", "Down"],
                            format_func=lambda x: {
                                "Up": "Up (ปกติ)",
                                "Flat": "Flat (ผิดปกติ)",
                                "Down": "Down (ผิดปกติมาก)"
                            }[x])

st.divider()

# --- Prediction ---
if st.button("🔍 ประเมินความเสี่ยง", type="primary", use_container_width=True):
    # สร้าง input DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp if resting_bp > 0 else np.nan,
        'Cholesterol': cholesterol if cholesterol > 0 else np.nan,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # แสดงผล
    st.markdown("### 📊 ผลการประเมิน")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.error(f"⚠️ **มีความเสี่ยงโรคหัวใจ**")
            st.metric("ความน่าจะเป็น", f"{probability[1]*100:.1f}%")
        else:
            st.success(f"✅ **ความเสี่ยงต่ำ**")
            st.metric("ความน่าจะเป็น (ปกติ)", f"{probability[0]*100:.1f}%")

    with result_col2:
        st.markdown("**รายละเอียด:**")
        st.write(f"- ความน่าจะเป็นที่จะเป็นโรคหัวใจ: **{probability[1]*100:.1f}%**")
        st.write(f"- ความน่าจะเป็นที่จะปกติ: **{probability[0]*100:.1f}%**")

    # Progress bar แสดง risk level
    st.markdown("**Risk Level:**")
    st.progress(probability[1])

# --- Disclaimer ---
st.divider()
st.caption("""
⚠️ **ข้อจำกัดสำคัญ:** ผลการประเมินนี้เป็นเพียงการคัดกรองเบื้องต้นด้วย Machine Learning
ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้ หากมีข้อสงสัยหรือมีอาการผิดปกติ
กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ
""")
