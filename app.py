import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

app = Flask(__name__)

# Check if model exists, if not train it
if not (os.path.exists('model/spam_classifier.pkl') and os.path.exists('model/vectorizer.pkl')):
    # Load and prepare the data
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df.columns = ['label', 'message']
    
    # Create feature vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Save the model and vectorizer
    os.makedirs('model', exist_ok=True)
    with open('model/spam_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
else:
    # Load the model and vectorizer
    with open('model/spam_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    if message:
        # Transform the message
        message_vec = vectorizer.transform([message])
        # Make prediction
        prediction = classifier.predict(message_vec)[0]
        # Get probability
        prob = classifier.predict_proba(message_vec)[0]
        probability = round(max(prob) * 100, 2)
        
        return render_template('index.html', 
                             prediction=prediction,
                             message=message,
                             probability=probability)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)