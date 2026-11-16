
# SMS Spam Detection Using Machine Learning

## 1. Project Overview

This project develops a machine-learning-based SMS Spam Detection System that classifies text messages into two categories:

- Ham (Legitimate Message)
- Spam (Unwanted or Fraudulent Message)

The system uses TF-IDF text representation, numerical feature extraction and machine-learning classification.

## 2. Team Members

| Name | Roll Number |
|---|---|
| Nishant Dalal | 24CSU291 |
| Devansh Gaur | 24CSU277 |
| Shreyash Bajpai | 24CSU261 |

## 3. Dataset

Dataset: UCI SMS Spam Collection

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains more than 5,500 SMS messages labelled as ham or spam.

## 4. Features

The system extracts the following numerical features:

| Feature | Description |
|---|---|
| message_length | Total characters |
| word_count | Total words |
| digit_count | Number of digits |
| uppercase_count | Number of uppercase characters |
| url_count | Number of URLs |

Text messages are also converted into TF-IDF features.

## 5. Machine Learning Algorithms

- Multinomial Naive Bayes
- Logistic Regression

Naive Bayes with tuned alpha was selected based on experimental F1-score.

## 6. Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI
- HTML, CSS and JavaScript
- Pytest
- Docker
- GitHub Actions

## 7. Project Structure

```text
SMS-Spam-Detection/
├── data/
├── model/
├── static/
├── tests/
├── .github/
│   └── workflows/
├── app.py
├── train.py
├── Dockerfile
├── requirements.txt
├── requirements-serve.txt
└── README.md
```

## 8. Running the Application

Install dependencies:

```bash
pip install -r requirements-serve.txt
```

Start the API:

```bash
uvicorn app:app --reload
```

Open:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## 9. Model Training

Run:

```bash
python train.py --nb_alpha 0.5 --max_iter 1000
```

The script compares three experiments and saves the selected model.

## 10. Testing

Run:

```bash
pytest tests/ -v
```

Five automated tests have passed during development.

## 11. API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| / | GET | Web interface |
| /health | GET | Health check |
| /predict | POST | Predict SMS category |

## 12. Experiment Tracking

MLflow is used to track model experiments and evaluation metrics, including accuracy, precision, recall and F1-score.

## 13. Continuous Integration

GitHub Actions automates dependency installation, model training and automated testing.

## 14. Future Improvements

- Improve spam recall.
- Evaluate additional machine-learning algorithms.
- Test on newer SMS datasets.
- Improve model monitoring and deployment.
- Add more advanced text-processing techniques.

## 15. Academic Project

Course: CSL236 – Introduction to AI and Machine Learning

Institution: The NorthCap University

Program: B.Tech Computer Science and Engineering
