# Fake News Detection

This project is a **Fake News Detection System** that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Real or Fake**. The system is built using **Flask** for the web interface and **Naïve Bayes** for classification.

## Features
- Preprocesses news articles using **NLTK** (stopword removal, stemming, lemmatization)
- Extracts features using **TF-IDF Vectorization**
- Classifies news as **Real or Fake** using **Multinomial Naïve Bayes**
- Flask-based web app for user input and predictions

## Technologies Used
- **Python**
- **Flask** (for the web interface)
- **Scikit-learn** (for ML model training)
- **NLTK** (for text preprocessing)
- **Joblib** (for model persistence)
- **Jupyter Notebook** (for experimentation)
- **GitHub** (for version control)

## Installation & Setup
### Clone the Repository
```sh
git clone https://github.com/LaibaSaleem/FakeNewsDetection.git
cd FakeNewsDetection
```

### Install dependencies
```sh
pip install -r requirements.txt
```  

### Run the app
```sh
python app.py
```
The web app will be available at http://127.0.0.1:5000/.
