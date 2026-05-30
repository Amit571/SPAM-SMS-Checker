📌 Project Title: Spam Message Classifier Web App
This project is a Flask-based web application that detects whether a message is Spam or Not Spam (Ham) using a Machine Learning model.

It uses the Naive Bayes algorithm (MultinomialNB) for classification.

The model is trained on a labeled SMS dataset (spam.csv) containing spam and non-spam messages.

Messages are preprocessed using TF-IDF Vectorization to convert text into numeric form.

Once trained, the model and vectorizer are saved using pickle for reuse.

When a user submits a message via the web form, it is processed and classified, and the prediction along with confidence (%) is displayed.

💡 Key Technologies Used:
Flask – Web framework for Python.

Pandas – Data handling and cleaning.

Scikit-learn – ML model training and text processing.

HTML (Jinja2) – Frontend templates.

Pickle – For model persistence.
