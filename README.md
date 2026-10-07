# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on customer demographics, services, contract information, and billing details.

## Project Overview

Customer churn is an important business problem.

If a company can identify customers who are likely to leave, it can take preventive actions such as:

- Offering discounts
- Providing better support
- Improving customer experience
- Offering personalized plans

This project builds an end-to-end machine learning system for customer churn prediction.

## Machine Learning Pipeline

Data
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Train/Test Split
↓
Model Training
↓
Model Evaluation
↓
Hyperparameter Tuning
↓
Model Saving
↓
FastAPI
↓
Streamlit
↓
Docker
↓
Deployment

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Streamlit
- Joblib
- Docker

## Machine Learning Models

The project experiments with:

- Logistic Regression
- Decision Tree
- Random Forest

Random Forest is tuned using GridSearchCV.

## Features

The model uses customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Contract
- Payment Method
- Monthly Charges
- Total Charges

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

Because customer churn datasets can be imbalanced, accuracy alone is not used to judge model performance.

## API

The project provides a FastAPI backend.

### Health Check

GET:

/health

### Prediction

POST:

/predict

Example response:

{
    "prediction": 1,
    "churn_probability": 0.82
}

Where:

0 = Customer is unlikely to churn

1 = Customer is likely to churn

## Running the Project

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Move into the project:

cd customer-churn-ml

Create virtual environment:

python3 -m venv venv

Activate:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Run FastAPI

uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs

## Run Streamlit

If the API is running on a different host port, set `API_URL` to match. For
example, when Docker publishes the API on port `8001`:

```bash
API_URL=http://127.0.0.1:8001/predict streamlit run app/streamlit_app.py
```

Without `API_URL`, Streamlit uses `http://127.0.0.1:8000/predict`.

streamlit run app/streamlit_app.py

## Docker

Build the Docker image:

docker build -t customer-churn-api .

Run the container:

docker run -p 8001:8000 customer-churn-api

Port `8001` is the host port; use `API_URL=http://127.0.0.1:8001/predict`
when starting Streamlit if port `8000` is already in use.

## Future Improvements

- Cloud deployment
- Model monitoring
- Data drift detection
- Automated retraining
- CI/CD pipeline
- Model versioning
- Authentication
- Database integration

## Author

Rajiv Mahato