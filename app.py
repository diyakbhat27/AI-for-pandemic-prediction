import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from io import StringIO

# Streamlit Page Configuration
st.set_page_config(page_title="Pandemic Prediction", layout="wide")
st.title("AI for Pandemic Prediction")

# File Uploaders
st.subheader("Step 1: Upload Required Files")
uploaded_csv = st.file_uploader("Upload the Cleaned Test Data CSV (cleaned_40_percent.csv)", type=["csv"])
uploaded_scaler = st.file_uploader("Upload the Scaler File (scaler.pkl)", type=["pkl"])
uploaded_model = st.file_uploader("Upload the Trained Model File (trained_model.pkl)", type=["pkl"])
uploaded_threshold = st.file_uploader("Upload the Impact Metric Threshold (impact_metric_threshold.txt)", type=["txt"])

# Proceed if all files are uploaded
if uploaded_csv and uploaded_scaler and uploaded_model and uploaded_threshold:
    try:
        # Load files
        test_data = pd.read_csv(uploaded_csv)
        scaler = joblib.load(uploaded_scaler)
        model = joblib.load(uploaded_model)
        saved_threshold = float(StringIO(uploaded_threshold.read().decode()).read())

        # Preprocess the data
        st.subheader("Step 2: Data Processing and Visualization")
        st.write("Dataset Preview:")
        st.dataframe(test_data.head())

        test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d-%m-%Y', errors='coerce')
        test_data = test_data.dropna(subset=['Date']).set_index('Date').sort_index()

        if 'Impact Metric' not in test_data.columns:
            test_data['Impact Metric'] = (
                0.5 * test_data['Confirmed'] +
                0.3 * test_data['Deaths'] -
                0.1 * test_data['Recovered'] +
                0.3 * test_data['Active']
            )

        # Scaling the features
        features = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        test_data[features] = scaler.transform(test_data[features])

        # Group by Date for visualization
        test_data_grouped = test_data.groupby(test_data.index)['Impact Metric'].mean()

        # Plot the graph
        st.subheader("Step 3: Result Visualization")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(test_data_grouped.index, test_data_grouped, color="blue", linewidth=2, label="Test Data Impact Metric")
        ax.axhline(y=saved_threshold, color='red', linestyle='--', linewidth=1.5, label="Threshold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Impact Metric")
        ax.set_title("Time vs Aggregated Impact Metric (Test Data)")
        ax.grid(True)

        # Add result message on the graph
        if test_data_grouped.max() > saved_threshold:
            ax.text(test_data_grouped.index[-1], saved_threshold + 5,
                    "Potential for Airborne Global Pandemic", color="red", fontsize=15, weight='bold')
            st.markdown("<h2 style='color:red'>WARNING: Potential for Airborne Global Pandemic</h2>", unsafe_allow_html=True)
        else:
            ax.text(test_data_grouped.index[-1], saved_threshold + 5,
                    "Low Risk for Airborne Global Pandemic", color="green", fontsize=15, weight='bold')
            st.markdown("<h2 style='color:green'>STATUS: Low Risk for Airborne Global Pandemic</h2>", unsafe_allow_html=True)

        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please upload all required files to proceed.")
