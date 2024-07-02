import os
import joblib
import pandas as pd
import streamlit as st
import shap
import streamlit.components.v1 as components

# 加载模型和SHAP解释器
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best_lgbm_model.pkl')
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

# 定义预测函数
def predict_risk(features):
    df = pd.DataFrame(features, index=[0])
    prediction = model.predict_proba(df)[:, 1]  # 获取正类（冠心病）的概率
    return prediction[0], df

# 定义解释函数
def explain_prediction(data):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    base_value = explainer.expected_value

    shap.initjs()

    if isinstance(shap_values, list):
        # 选择第一个输出
        shap_values = shap_values[0]
        base_value = base_value[0]

    return shap.force_plot(base_value, shap_values, data, matplotlib=True)

# Streamlit 应用
st.title('Risk prediction of coronary heart disease in young adults with hypertension')

# 创建输入表单
average_drink_per_day = st.number_input('Average drink per day', min_value=1, max_value=8, value=1)
type_of_work = st.selectbox('Type of work', ['Working at a job or business', 'Not working at a job or business'])
health_of_teeth_gums = st.selectbox('health of teeth and gums', ['Very good', 'Good', 'Fair', 'Poor'])
high_cholesterol_level = st.radio('Whether high cholesterol levels', ['Yes', 'No'])
arthritis = st.radio('arthritis', ['Yes', 'No'])
tobacco_use = st.radio('Once smoked a hundred cigarettes', ['Yes', 'No'])
standing_height = st.number_input('Standing Height', min_value=0.0, value=160.0)
relative_asthma = st.radio('Close relative had asthma', ['Yes', 'No'])
recreational_activities = st.selectbox('Moderate recreational activities', ['Low', 'Moderate', 'High'])
heaviest_weight_age = st.number_input('Age when heaviest weight', min_value=0, value=30)
depression = st.radio('Depression', ['0', '1', '2', '3', '4', '5'])
sleep_hours = st.number_input('Sleep hours', min_value=0.0, value=7.0)
sleep_disorders = st.radio('Sleep Disorders', ['Yes', 'No'])
night_urination = st.number_input('How many times urinate in night', min_value=1, max_value=5, value=1)
total_cholesterol = st.number_input('Total cholesterol', min_value=0.0, value=5.0)
mercury = st.number_input('Mercury', min_value=0.0, value=4.0)
monocyte_number = st.number_input('Monocyte number', min_value=0.0, value=0.5)
lead = st.number_input('Lead', min_value=0.0, value=0.05)
platelet_count = st.number_input('Platelet count (%)', min_value=0.0, value=250.0)
hdl = st.number_input('HDL', min_value=0.0, value=50.0)
rdw = st.number_input('Red cell distribution width (%)', min_value=0.0, value=12.0)
potassium = st.number_input('Potassium', min_value=0.0, value=4.0)
alt = st.number_input('ALT', min_value=0.0, value=10.0)
uric_acid = st.number_input('Uric acid', min_value=0.0, value=300.0)
lymphocyte_percent = st.number_input('Lymphocyte percent (%)', min_value=0.0, value=30.0)
cadmium = st.number_input('Cadmium', min_value=0.0, value=3.0)

# 映射牙齿和牙龈健康的选择到模型中的数值
health_mapping = {
    'Very good': 1,
    'Good': 2,
    'Fair': 3,
    'Poor': 4
}

# 构建特征字典
features = {
    'Average drink per day': average_drink_per_day,
    'Type of work': 1 if type_of_work == 'Working at a job or business' else 4,
    'Health of teeth and gums': health_mapping[health_of_teeth_gums],
    'High cholesterol level': 1 if high_cholesterol_level == 'Yes' else 0,
    'Total cholesterol': total_cholesterol,
    'Arthritis': 1 if arthritis == 'Yes' else 0,
    'Tobacco Use': 1 if tobacco_use == 'Yes' else 0,
    'Standing Height': standing_height,
    'Close relative had asthma': 1 if relative_asthma == 'Yes' else 0,
    'Moderate recreational activities': 1 if recreational_activities == 'Low' else (
        2 if recreational_activities == 'Moderate' else 3),
    'Age when heaviest weight': heaviest_weight_age,
    'Depression': int(depression),
    'Sleep hours': sleep_hours,
    'Sleep Disorders': 1 if sleep_disorders == 'Yes' else 0,
    'How many times urinate in night': night_urination,
    'Mercury': mercury,
    'Monocyte number': monocyte_number,
    'Lead': lead,
    'Platelet count (%)': platelet_count,
    'HDL': hdl,
    'Red cell distribution width (%)': rdw,
    'Potassium': potassium,
    'ALT': alt,
    'Uric acid': uric_acid,
    'Lymphocyte percent (%)': lymphocyte_percent,
    'Cadmium': cadmium
}

# 预测和解释
if st.button('prediction'):
    prediction, df = predict_risk(features)
    st.write(f"The predicted probability of coronary heart disease is: {prediction:.2%}")
    explanation = explain_prediction(df)
    st.pyplot(explanation)




