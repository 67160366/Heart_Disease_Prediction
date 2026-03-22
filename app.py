import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="CVD Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('cardio_model.pkl')

model = load_model()

# --- Feature Importance Data ---
FEATURE_NAMES = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi',
                 'cholesterol', 'gluc', 'active']
FEATURE_LABELS = {
    'ap_hi': 'ความดัน Systolic',
    'age': 'อายุ',
    'cholesterol': 'คอเลสเตอรอล',
    'bmi': 'BMI',
    'weight': 'น้ำหนัก',
    'ap_lo': 'ความดัน Diastolic',
    'height': 'ส่วนสูง',
    'gluc': 'กลูโคส',
    'active': 'ออกกำลังกาย',
}

# --- Header ---
st.title("❤️ Cardiovascular Disease Risk Prediction")
st.markdown("""
ระบบประเมินความเสี่ยงโรคหลอดเลือดหัวใจเบื้องต้นด้วย Machine Learning
กรอกข้อมูลด้านล่างเพื่อประเมินความเสี่ยง
""")

st.divider()

# --- Input Form ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ข้อมูลทั่วไป")
    age = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=50, step=1,
                           help="อายุของผู้รับการประเมิน เป็นปัจจัยสำคัญอันดับต้นๆ ที่ส่งผลต่อความเสี่ยง CVD")
    height = st.number_input("ส่วนสูง (cm)", min_value=100, max_value=220, value=165, step=1,
                              help="ส่วนสูงใช้คำนวณ BMI ร่วมกับน้ำหนัก")
    weight = st.number_input("น้ำหนัก (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5,
                              help="น้ำหนักใช้คำนวณ BMI ซึ่งเป็นตัวบ่งชี้ความเสี่ยงโรคหัวใจ")

with col2:
    st.subheader("ผลตรวจความดัน")
    ap_hi = st.number_input("ความดัน Systolic (mm Hg)", min_value=50, max_value=300, value=120, step=1,
                             help="ค่าตัวบน เช่น 120/80 → กรอก 120 | ค่าปกติ: 90-120 mm Hg — เป็นปัจจัยที่สำคัญที่สุดในการทำนาย CVD")
    ap_lo = st.number_input("ความดัน Diastolic (mm Hg)", min_value=30, max_value=200, value=80, step=1,
                             help="ค่าตัวล่าง เช่น 120/80 → กรอก 80 | ค่าปกติ: 60-80 mm Hg")

    # --- Input Validation: Blood Pressure ---
    if ap_hi <= ap_lo:
        st.warning("⚠️ ค่า Systolic ควรสูงกว่า Diastolic — กรุณาตรวจสอบอีกครั้ง")
    if ap_hi > 200:
        st.info("ℹ️ ค่า Systolic สูงมาก (>200) — กรุณาตรวจสอบว่ากรอกถูกต้อง")
    if ap_lo < 40:
        st.info("ℹ️ ค่า Diastolic ต่ำมาก (<40) — กรุณาตรวจสอบว่ากรอกถูกต้อง")

with col3:
    st.subheader("ผลตรวจ & พฤติกรรม")
    cholesterol = st.selectbox("ระดับคอเลสเตอรอล",
                                options=[1, 2, 3],
                                format_func=lambda x: {
                                    1: "1 — ปกติ",
                                    2: "2 — สูงกว่าปกติ",
                                    3: "3 — สูงมาก"
                                }[x],
                                help="ระดับคอเลสเตอรอลในเลือด — คอเลสเตอรอลสูงเป็นปัจจัยเสี่ยงสำคัญของโรคหลอดเลือดหัวใจ")
    gluc = st.selectbox("ระดับกลูโคส",
                         options=[1, 2, 3],
                         format_func=lambda x: {
                             1: "1 — ปกติ",
                             2: "2 — สูงกว่าปกติ",
                             3: "3 — สูงมาก"
                         }[x],
                         help="ระดับน้ำตาลกลูโคสในเลือด — ระดับสูงอาจบ่งบอกถึงเบาหวาน ซึ่งเพิ่มความเสี่ยง CVD")
    active = st.selectbox("ออกกำลังกายเป็นประจำ?", options=[1, 0],
                           format_func=lambda x: "ออก" if x == 1 else "ไม่ออก",
                           help="การออกกำลังกายสม่ำเสมอช่วยลดความเสี่ยงโรคหัวใจได้อย่างมีนัยสำคัญ")

st.divider()

# --- BMI Preview ---
bmi = weight / (height / 100) ** 2
bmi_status = "ปกติ" if 18.5 <= bmi <= 25 else "เกินเกณฑ์" if bmi > 25 else "ต่ำกว่าเกณฑ์"

col_bmi1, col_bmi2, _ = st.columns(3)
with col_bmi1:
    st.metric("BMI (คำนวณอัตโนมัติ)", f"{bmi:.1f}", delta=bmi_status,
              delta_color="off" if bmi_status == "ปกติ" else "inverse")

# --- Input Validation: BMI ---
if bmi > 50:
    st.warning("⚠️ BMI สูงผิดปกติ (>50) — กรุณาตรวจสอบส่วนสูงและน้ำหนัก")
elif bmi < 15:
    st.warning("⚠️ BMI ต่ำผิดปกติ (<15) — กรุณาตรวจสอบส่วนสูงและน้ำหนัก")

st.divider()

# --- Prediction ---
if st.button("🔍 ประเมินความเสี่ยง", type="primary", use_container_width=True):

    # ตรวจสอบค่าที่ไม่สมเหตุสมผล
    has_warning = False
    if ap_hi <= ap_lo:
        st.error("❌ ไม่สามารถประเมินได้ — ค่า Systolic ต้องสูงกว่า Diastolic")
        has_warning = True

    if not has_warning:
        # สร้าง input DataFrame (9 features — ตัด gender, smoke, alco ออกแล้ว)
        input_data = pd.DataFrame([{
            'age': age,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'bmi': bmi,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'active': active,
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        # --- ผลการประเมิน ---
        st.markdown("### 📊 ผลการประเมิน")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ **มีความเสี่ยงโรคหลอดเลือดหัวใจ**")
                st.metric("ความน่าจะเป็น (CVD)", f"{probability[1]*100:.1f}%")
            else:
                st.success("✅ **ความเสี่ยงต่ำ**")
                st.metric("ความน่าจะเป็น (ปกติ)", f"{probability[0]*100:.1f}%")

        with result_col2:
            st.markdown("**รายละเอียด:**")
            st.write(f"- ความน่าจะเป็นที่จะเป็น CVD: **{probability[1]*100:.1f}%**")
            st.write(f"- ความน่าจะเป็นที่จะปกติ: **{probability[0]*100:.1f}%**")
            st.write(f"- BMI: **{bmi:.1f}** ({bmi_status})")

        # Risk level bar
        st.markdown("**Risk Level:**")
        st.progress(probability[1])

        # --- Feature Importance ---
        st.divider()
        st.markdown("### 🔬 ปัจจัยที่ส่งผลต่อการทำนาย (Feature Importance)")
        st.caption("แสดงว่า model ให้น้ำหนักกับปัจจัยใดมากที่สุดในการตัดสินใจ")

        importances = model.named_steps['classifier'].feature_importances_
        feat_imp = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'importance': importances
        })
        feat_imp['label'] = feat_imp['feature'].map(FEATURE_LABELS)
        feat_imp = feat_imp.sort_values('importance', ascending=True)

        # Horizontal bar chart
        st.bar_chart(
            feat_imp.set_index('label')['importance'],
            horizontal=True,
            color='#e74c3c'
        )

        # Top 3 factors explanation
        top3 = feat_imp.nlargest(3, 'importance')
        st.markdown("**ปัจจัยสำคัญ 3 อันดับแรก:**")
        for _, row in top3.iterrows():
            pct = row['importance'] * 100
            st.write(f"- **{row['label']}** — มีอิทธิพล {pct:.1f}% ต่อการตัดสินใจของ model")

# --- Disclaimer ---
st.divider()
st.caption("""
⚠️ **ข้อจำกัดสำคัญ:** ผลการประเมินนี้เป็นเพียงการคัดกรองเบื้องต้นด้วย Machine Learning
ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้ หากมีข้อสงสัยหรือมีอาการผิดปกติ
กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ
""")
