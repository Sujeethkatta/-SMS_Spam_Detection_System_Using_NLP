# SMS Spam Detection System

**Repository summary**

This project implements an SMS Spam Detection System using Natural Language Processing (NLP) and a classical machine‑learning classifier. It contains data, training notebooks, a saved TF‑IDF vectorizer, a saved trained model, and a small Flask app to demo predictions.

---

## Files in this repository

* `spam.csv` — dataset of SMS messages labeled as `spam` or `ham` used for training and evaluation.
* `SMS_Spam_Detection_System_Using_NLP_P1.ipynb` — notebook showing data cleaning, feature extraction (TF‑IDF), training the classifier, evaluation and saving the model/vectorizer.
* `sms-spam-detection.ipynb` — additional notebook (exploratory analysis or alternative model experiments).
* `tfidf_vectorizer.pkl` — serialized TF‑IDF vectorizer used to transform raw messages into numeric features.
* `spam_classifier_model.pkl` — serialized trained classifier (e.g., Logistic Regression / Naive Bayes / SVM) used for prediction.
* `app.py` — lightweight demo web application (Flask) that loads the vectorizer and model to make live predictions.
* `Project_Report_Template.pdf` — project report or template describing objectives, methodology and results.

---

## Quick start

### Requirements

* Python 3.8+
* Recommended packages (create a `requirements.txt` with these):

```
pandas
numpy
scikit-learn
flask
joblib
nltk
jinja2
```

**Note:** If the notebooks use other packages (e.g., seaborn, matplotlib), install them as needed.

### Install

1. Create a virtual environment and activate it:

```bash
python -m venv venv
# windows
venv\Scripts\activate
# mac / linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

(If you don't have `requirements.txt`, install the packages listed above.)

### Run the demo app

1. Ensure `spam_classifier_model.pkl` and `tfidf_vectorizer.pkl` are in the repository root (or update `app.py` paths).
2. Start the Flask app:

```bash
python app.py
```

3. Open your browser at `http://127.0.0.1:5000/` (or the host/port printed by Flask) and enter a message to see whether it's classified as **spam** or **ham**.

---

## How to retrain the model (outline)

If you want to retrain the model from scratch using the included notebooks:

1. Open `SMS_Spam_Detection_System_Using_NLP_P1.ipynb` in Jupyter Notebook / JupyterLab.
2. Run the cells to load `spam.csv`, preprocess text (lowercase, remove punctuation, tokenization, stopword removal, optional lemmatization), and split into train/test sets.
3. Fit a `TfidfVectorizer` on training text and transform the train/test sets.
4. Train a classifier (e.g., `LogisticRegression`, `MultinomialNB`, `RandomForestClassifier`) and evaluate using accuracy, precision, recall, F1, and a confusion matrix.
5. Save the trained model and vectorizer:

```python
from joblib import dump

dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
dump(model, 'spam_classifier_model.pkl')
```

6. Replace the `.pkl` files in the repo or update `app.py` to point to the new files.

---

## `app.py` (expected behavior)

`app.py` should perform the following:

1. Load `tfidf_vectorizer.pkl` and `spam_classifier_model.pkl` at startup.
2. Expose a web form (or API endpoint) that accepts a text message.
3. Transform the incoming text using the vectorizer and predict with the classifier.
4. Return the predicted label (`spam` / `ham`) and optionally the probability score.

Example API usage (if `app.py` includes a `/predict` POST endpoint):

```bash
curl -X POST -H "Content-Type: application/json" -d '{"message":"Free entry in 2 a wkly comp..."}' http://127.0.0.1:5000/predict
```

Response example (JSON):

```json
{ "label": "spam", "probability": 0.98 }
```

---

## Recommended improvements and ideas

* Add model versioning (include a `models/` folder with timestamps or version numbers).
* Add unit tests for preprocessing functions and the prediction endpoint.
* Add a `requirements.txt` and a `Procfile` for easy deployment (Heroku/Gunicorn or similar).
* Use a simple Dockerfile for reproducible deployments.
* Expand preprocessing (lemmatization, better handling of emojis and URLs).
* Add rate‑limiting / basic auth if exposing the API publicly.

---

## License

Choose a license for the repo (for example, MIT). Add a `LICENSE` file.

---

## Contact / Author

Project maintained by **Sujeethkatta**.

If you want changes to this README or want me to generate a `requirements.txt`, a `Dockerfile`, or update `app.py` to a production-ready Flask app, tell me what you prefer and I will prepare it.
