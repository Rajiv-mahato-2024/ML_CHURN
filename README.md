# ML Churn

This project trains a small customer-churn model using a synthetic dataset and exposes prediction interfaces.

## Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.train
```

## Tune the model

```bash
python -m src.tune_model
```

## Run the FastAPI service

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at http://127.0.0.1:8000/docs.

The `/predict` endpoint accepts `tenure`, `monthly_charges`, `total_charges`,
`contract_type`, `internet_service`, and `support`.

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Structure

- `src/data_cleaning.py`: dataset generation and data cleaning
- `src/feature_engineering.py`: preprocessing for numeric and categorical fields
- `src/train.py`: training and model saving logic
- `src/tune_model.py`: cross-validated logistic-regression tuning
- `src/predict.py`: prediction helper for a single record
- `app/main.py`: FastAPI prediction service
- `app/streamlit_app.py`: Streamlit prediction interface
