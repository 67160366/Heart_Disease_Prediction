# Cardiovascular Disease Prediction — ML Deployment Project

ระบบประเมินความเสี่ยงโรคหลอดเลือดหัวใจ (CVD) เบื้องต้นด้วย Machine Learning พร้อม deploy เป็น web application ด้วย Streamlit

## ปัญหาที่ต้องการแก้ไข
โรคหลอดเลือดหัวใจ (Cardiovascular Disease) เป็นสาเหตุการเสียชีวิตอันดับ 1 ของโลก คิดเป็นประมาณ 17.9 ล้านคนต่อปี โปรเจคนี้สร้าง ML model เพื่อ **คัดกรองเบื้องต้น** จากข้อมูลสุขภาพพื้นฐาน เช่น อายุ ความดันโลหิต คอเลสเตอรอล และพฤติกรรมสุขภาพ

## Dataset
- **ที่มา**: [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (Kaggle)
- **ขนาด**: 70,000 ตัวอย่าง, 12 features + 1 target
- **Task**: Binary Classification (เป็น CVD / ไม่เป็น)

### Features
| Feature | คำอธิบาย | ประเภท |
|---------|----------|--------|
| age | อายุ (วัน → แปลงเป็นปี) | Numerical |
| gender | เพศ (1=หญิง, 2=ชาย) | Categorical |
| height | ส่วนสูง (cm) | Numerical |
| weight | น้ำหนัก (kg) | Numerical |
| ap_hi | ความดัน Systolic (ตัวบน) | Numerical |
| ap_lo | ความดัน Diastolic (ตัวล่าง) | Numerical |
| cholesterol | ระดับคอเลสเตอรอล (1=ปกติ, 2=สูง, 3=สูงมาก) | Categorical |
| gluc | ระดับกลูโคส (1=ปกติ, 2=สูง, 3=สูงมาก) | Categorical |
| smoke | สูบบุหรี่ (0=ไม่, 1=ใช่) | Binary |
| alco | ดื่มแอลกอฮอล์ (0=ไม่, 1=ใช่) | Binary |
| active | ออกกำลังกาย (0=ไม่, 1=ใช่) | Binary |
| **cardio** | **เป็นโรค CVD (0=ไม่, 1=ใช่) — Target** | **Binary** |

## วิธีการ
1. **EDA** — วิเคราะห์ distribution, outliers ทางการแพทย์, correlation
2. **Preprocessing** — ลบ outliers + สร้าง BMI + Pipeline (StandardScaler + passthrough categorical)
3. **Model Comparison** — เปรียบเทียบ 4 algorithms (Logistic Regression, Random Forest, Gradient Boosting, KNN) ด้วย cross-validation
4. **Hyperparameter Tuning** — GridSearchCV บน Gradient Boosting
5. **Deployment** — Streamlit web app พร้อม feature importance visualization

## ผลลัพธ์
| Model | Accuracy | F1 | ROC AUC |
|-------|----------|-----|---------|
| Logistic Regression (baseline) | ~0.72 | ~0.72 | ~0.78 |
| **Gradient Boosting (tuned)** | **~0.73** | **~0.73** | **~0.80** |

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
├── cardio_train.csv            # Dataset (70k rows)
├── heart_disease_project.ipynb # Notebook (EDA + Model)
├── app.py                      # Streamlit web app
├── heart_disease_model.pkl     # Trained model (สร้างจาก notebook)
├── requirements.txt            # Dependencies
├── .gitignore
└── README.md
```

## ข้อจำกัด
- ผลการประเมินเป็นเพียงการคัดกรองเบื้องต้น **ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์**
- Dataset มาจาก medical examination records ซึ่งอาจมี bias จากประชากรที่ศึกษา
