# AI for Pandemic Prediction

## 🌍 Overview
*AI for Pandemic Prediction* is a machine learning-based mini-project developed to predict the future impact of pandemics using historical COVID-19 data. By analyzing parameters such as confirmed cases, deaths, recoveries, and active cases, the model aims to forecast pandemic trends and assist in decision-making for public health preparedness.

---

## 📌 Objectives
- Predict the severity of future pandemics using machine learning.
- Analyze COVID-19 data to detect trends and possible future outbreaks.
- Provide data-driven insights for early preparedness.

---

## 🧠 Technologies & Tools Used
- *Python* – Programming language used for implementation
- *Streamlit* – To develop the interactive web-based interface
- *Pandas & NumPy* – For data manipulation and preprocessing
- *Matplotlib & Seaborn* – For data visualization
- *Scikit-learn* – For implementing machine learning models
- *COVID-19 Dataset* – Sourced for training and analysis

---

## 📈 Workflow

1. *Data Collection*  
   - Dataset based on COVID-19 cases including confirmed, deaths, recovered, and active cases.

2. *Data Preprocessing*  
   - Handling missing values and formatting the data.

3. *Model Building*  
   - A simple classification model is trained to predict the severity of pandemic-like situations based on input metrics.

4. *Prediction*  
   - Users can input new data through the Streamlit interface to predict whether the situation may lead to a global pandemic.

---

## 💻 How to Run the Project

Make sure you have Python and Streamlit installed. Then run the following command in your terminal:

```bash
streamlit run app.py 
