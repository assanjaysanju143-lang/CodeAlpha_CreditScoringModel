# Credit Scoring Model

This project is a simple machine learning and Streamlit application for credit risk prediction.
It was built as part of a CodeAlpha internship project and includes both model training logic and
an interactive web app.

## Features

- Predicts whether an applicant is likely to be safe or risky
- Uses logistic regression with preprocessing in a reusable pipeline
- Handles missing values in the dataset
- Shows model metrics and simple visual insights
- Stores previous predictions in `history.csv`

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib

## Files

- `credit_model.py`: data loading, preprocessing, training, and evaluation
- `app.py`: Streamlit interface for prediction and analytics
- `german_credit_data.csv`: source dataset
- `history.csv`: saved prediction history

## How To Run

Install the required packages, then start the app:

```bash
streamlit run app.py
```

To run the model training script directly:

```bash
python credit_model.py
```

## Current Target Logic

The dataset in this repository does not include a built-in credit risk label.
For demonstration purposes, this project derives a temporary target:

- `Safe (1)` when `Credit amount <= 5000`
- `Risky (0)` when `Credit amount > 5000`

This makes the project useful for learning and UI demonstration, but it is not a real production credit scoring label.
