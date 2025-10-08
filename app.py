import streamlit as st
import pandas as pd
import joblib
import requests


st.set_page_config(
    page_icon="🧠",
    layout="centered",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide deploy button and menu
hide_menu_style = """
    <style>
    .stBaseButton-header {display:none;}
    [data-testid="stBaseButton-header"] {display:none !important;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- Load model and encoders ---
model = joblib.load("alzheimers_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


st.title("🧠 Alzheimer's Disease Prediction")
st.write("Provide the following patient details:")


# --- Get query params using new Streamlit API ---
query_params = st.query_params

API_ENDPOINT = query_params.get("url", "")
access_token = query_params.get("accessToken", "")

if isinstance(API_ENDPOINT, list):
    API_ENDPOINT = API_ENDPOINT[0] if API_ENDPOINT else ""
if isinstance(access_token, list):
    access_token = access_token[0] if access_token else ""

# --- Helper to safely fetch encoder classes ---
def get_encoder_classes(column_name, default=["Yes", "No"]):
    """Return label encoder classes or fallback."""
    if column_name in label_encoders:
        return label_encoders[column_name].classes_
    else:
        st.warning(f"⚠️ Missing encoder for '{column_name}', using default options.")
        return default

# --- Collect user input ---
def user_input():
    data = {}

    # Demographics
    st.subheader("📋 Demographics")
    col1, col2 = st.columns(2)
    with col1:
        data["Country"] = st.selectbox("Country", get_encoder_classes("Country"))
        data["Age"] = st.slider("Age", 40, 100, 65)
        data["Gender"] = st.selectbox("Gender", get_encoder_classes("Gender"))
    with col2:
        data["Education Level"] = st.slider("Education Level (years)", 0, 20, 12)
        data["BMI"] = st.slider("BMI", 10.0, 50.0, 25.0)
        data["Employment Status"] = st.selectbox("Employment Status", get_encoder_classes("Employment Status"))

    # Lifestyle
    st.subheader("🏃 Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        data["Physical Activity Level"] = st.selectbox("Physical Activity", get_encoder_classes("Physical Activity Level"))
        data["Smoking Status"] = st.selectbox("Smoking Status", get_encoder_classes("Smoking Status"))
        data["Alcohol Consumption"] = st.selectbox("Alcohol Consumption", get_encoder_classes("Alcohol Consumption"))
    with col4:
        data["Dietary Habits"] = st.selectbox("Dietary Habits", get_encoder_classes("Dietary Habits"))
        data["Sleep Quality"] = st.selectbox("Sleep Quality", get_encoder_classes("Sleep Quality"))
        data["Air Pollution Exposure"] = st.selectbox("Air Pollution Exposure", get_encoder_classes("Air Pollution Exposure"))

    # Medical History
    st.subheader("🏥 Medical History")
    col5, col6 = st.columns(2)
    with col5:
        data["Diabetes"] = st.selectbox("Diabetes", get_encoder_classes("Diabetes"))
        data["Hypertension"] = st.selectbox("Hypertension", get_encoder_classes("Hypertension"))
        data["Cholesterol Level"] = st.selectbox("Cholesterol Level", get_encoder_classes("Cholesterol Level"))
    with col6:
        # handle apostrophe variant safely
        fam_col = "Family History of Alzheimer’s" if "Family History of Alzheimer’s" in label_encoders else "Family History of Alzheimer's"
        data[fam_col] = st.selectbox("Family History", get_encoder_classes(fam_col))
        data["Genetic Risk Factor (APOE-ε4 allele)"] = st.selectbox("APOE-ε4 allele", get_encoder_classes("Genetic Risk Factor (APOE-ε4 allele)"))
        data["Cognitive Test Score"] = st.slider("Cognitive Test Score", 0, 100, 50)

    # Mental & Social
    st.subheader("🧘 Mental & Social Health")
    col7, col8 = st.columns(2)
    with col7:
        data["Depression Level"] = st.selectbox("Depression Level", get_encoder_classes("Depression Level"))
        data["Stress Levels"] = st.selectbox("Stress Levels", get_encoder_classes("Stress Levels"))
        data["Social Engagement Level"] = st.selectbox("Social Engagement", get_encoder_classes("Social Engagement Level"))
    with col8:
        data["Marital Status"] = st.selectbox("Marital Status", get_encoder_classes("Marital Status"))
        data["Income Level"] = st.selectbox("Income Level", get_encoder_classes("Income Level"))
        data["Urban vs Rural Living"] = st.selectbox("Urban vs Rural Living", get_encoder_classes("Urban vs Rural Living"))

    return pd.DataFrame([data])


# --- Collect user inputs ---
input_df = user_input()
original_input_data = input_df.iloc[0].to_dict()

# --- Predict button ---
if st.button("🔍 Predict", use_container_width=True):
    try:
        encoded_df = input_df.copy()

        # Encode categorical inputs
        for col in encoded_df.columns:
            if col in label_encoders:
                encoded_df[col] = label_encoders[col].transform(encoded_df[col])
            else:
                # Only apply Yes/No mapping to known categorical columns
                if col in ["Diabetes", "Hypertension", "Family History of Alzheimer's", 
                          "Genetic Risk Factor (APOE-ε4 allele)"]:
                    encoded_df[col] = encoded_df[col].map({"Yes": 1, "No": 0}).fillna(0)
                # For numerical columns, keep as-is

        encoded_df.columns = [c.replace("'", "'").replace(""", '"').replace(""", '"') 
                             for c in encoded_df.columns]

        model_features = list(model.feature_names_in_)
        encoded_df = encoded_df.reindex(columns=model_features, fill_value=0)

        # --- Predict ---
        proba = model.predict_proba(encoded_df)[0]
        prediction = model.predict(encoded_df)[0]
        label = target_encoder.inverse_transform([prediction])[0]
        
        confidence = proba[1]  # Always index 1 for consistency
        
        # --- Display results ---
        st.subheader("🧾 Prediction Results:")
        
        # Check the predicted label to determine result
        label_str = str(label).lower()
        if "yes" in label_str or "positive" in label_str or "high" in label_str:
            st.error(f"⚠️ The model predicts: **{label}** ({confidence*100:.1f}% probability of Alzheimer's)")
            result = "Positive"
        else:
            st.success(f"✅ The model predicts: **{label}** ({confidence*100:.1f}% probability of Alzheimer's)")
            result = "Negative"

        with st.expander("📊 View Detailed Probabilities"):
            for idx, prob in enumerate(proba):
                class_label = target_encoder.inverse_transform([idx])[0]
                st.write(f"{class_label}: {prob*100:.2f}%")

        # --- Push to API ---
        if access_token and API_ENDPOINT:
            payload = {
                "model_type": "alzheimer_disease",
                "inputs": original_input_data,
                "prediction": result,
                "confidence": float(confidence)
            }
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            st.info("📤 Uploading prediction to your dashboard...")

            try:
                response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=10)
                if response.status_code in (200, 201):
                    st.success("✅ Prediction saved to your dashboard successfully!")
                else:
                    print(f"API responded with status {response.status_code}: {response.text}")
            except requests.exceptions.Timeout:
                print("Request timed out. Please check your connection.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Failed to connect to the endpoint.")
            except Exception as e:
                st.error("❌ Failed to upload prediction.")
        else:
            st.info("ℹ️ No API endpoint or access token provided; skipping upload.")

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write("Please ensure all inputs are valid and try again.")
        
# --- Footer ---
st.markdown("---")
st.caption("💡 This prediction is for informational purposes only. Please consult a healthcare professional for medical advice.")
