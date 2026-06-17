🌿 AQI Prediction Dashboard

A Machine Learning-powered Air Quality Index (AQI) Prediction Dashboard built using Python, Streamlit, Scikit-Learn, Plotly, and Twilio. The application enables users to analyze air quality data, visualize pollution trends, evaluate model performance, and predict AQI in real time through an interactive web interface.

🚀 Features
📊 Interactive AQI analytics dashboard
🤖 Random Forest Regression model for AQI prediction
📈 Exploratory Data Analysis (EDA)
🔍 Correlation and feature importance analysis
📉 Model evaluation using MAE, MSE, RMSE, and R² Score
🌍 City-wise AQI comparison
⚡ Real-time AQI prediction from pollutant inputs
📱 Twilio SMS alerts for hazardous AQI levels
🌙 Modern Dark Mode UI
📂 CSV dataset upload support
🛠️ Tech Stack
Python
Streamlit
Pandas
NumPy
Scikit-Learn
Plotly
Matplotlib
Seaborn
Twilio API
📋 Dataset Features

The model predicts AQI using:

PM2.5
PM10
NO
NO₂
NOx
NH₃
CO
SO₂
O₃
Benzene
Toluene
Xylene
🤖 Machine Learning Workflow
Dataset Upload
Data Cleaning & Preprocessing
Missing Value Imputation
Feature Selection
Train-Test Split (80:20)
Random Forest Model Training
Performance Evaluation
AQI Prediction
SMS Alert Generation
📊 Model Evaluation Metrics
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score
Prediction Accuracy


📸 Dashboard Modules
🏠 Dashboard
AQI statistics
Feature importance visualization
AQI distribution analysis
City-wise AQI comparison
🔍 Data Explorer
Dataset preview
Missing value heatmap
Descriptive statistics
Correlation matrix
📡 Model Performance
Actual vs Predicted AQI
Residual analysis
Feature importance explorer
Performance metrics
⚡ Predict AQI
Custom pollutant input
Real-time AQI prediction
AQI gauge visualization
Health impact assessment
SMS alert system
📱 AQI Alert System

When predicted AQI exceeds 300, the application automatically sends an SMS alert using Twilio to notify users about hazardous air quality conditions.

🎯 Future Enhancements
Weather-based AQI forecasting
Deep Learning models (LSTM, XGBoost)
AQI prediction for multiple cities
Real-time AQI API integration
Deployment on AWS/Azure
