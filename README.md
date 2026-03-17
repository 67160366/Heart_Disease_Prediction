# ❤️ Heart Disease Prediction — ML Deployment Project

ระบบประเมินความเสี่ยงโรคหัวใจเบื้องต้นด้วย Machine Learning พร้อม deploy เป็น web application ด้วย Streamlit

## 📋 สารบัญ
- [ปัญหาที่ต้องการแก้ไข](#ปัญหาที่ต้องการแก้ไข)
- [Dataset](#dataset)
- [วิธีการ](#วิธีการ)
- [ผลลัพธ์](#ผลลัพธ์)
- [การใช้งาน](#การใช้งาน)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)

## ปัญหาที่ต้องการแก้ไข
โรคหัวใจเป็นสาเหตุการเสียชีวิตอันดับ 1 ของโลก โปรเจคนี้สร้าง ML model เพื่อ **คัดกรองเบื้องต้น** จากข้อมูลทางการแพทย์พื้นฐาน เช่น อายุ ความดันโลหิต คอเลสเตอรอล ผล ECG

## Dataset
- **ที่มา**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle)
- **ขนาด**: 918 ตัวอย่าง, 11 features + 1 target
- **Task**: Binary Classification (เป็นโรคหัวใจ / ไม่เป็น)

## วิธีการ
1. **EDA** — วิเคราะห์ distribution, missing values (Cholesterol=0), correlation
2. **Preprocessing** — Pipeline: median imputation + StandardScaler + OneHotEncoder
3. **Model Comparison** — เปรียบเทียบ 5 algorithms ด้วย 5-fold CV
4. **Hyperparameter Tuning** — GridSearchCV บน Gradient Boosting
5. **Deployment** — Streamlit web app

## ผลลัพธ์
| Model | Accuracy | F1 | ROC AUC |
|-------|----------|-----|---------|
| Logistic Regression (baseline) | ~0.85 | ~0.87 | ~0.92 |
| **Gradient Boosting (tuned)** | **~0.87** | **~0.89** | **~0.93** |

*(ค่าจะอัปเดตหลังรัน notebook)*

## การใช้งาน

### ติดตั้ง
```bash
pip install -r requirements.txt
```

### รัน Notebook
เปิด `heart_disease_project.ipynb` ใน Jupyter Notebook แล้วรันทุก cell ตามลำดับ

### รัน Streamlit App
```bash
streamlit run app.py
```

## โครงสร้างโปรเจค
```
DataSci/
├── heart.csv                    # Dataset
├── heart_disease_project.ipynb  # Notebook (EDA + Model)
├── app.py                       # Streamlit web app
├── heart_disease_model.pkl      # Trained model (สร้างจาก notebook)
├── requirements.txt             # Dependencies
├── .gitignore
└── README.md
```

## ⚠️ ข้อจำกัด
- ผลการประเมินเป็นเพียงการคัดกรองเบื้องต้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์**
- Dataset มาจากหลายแหล่ง อาจมี bias จากประชากรที่ศึกษา
