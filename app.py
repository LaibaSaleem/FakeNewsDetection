from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained Naïve Bayes model and TF-IDF vectorizer
nb_model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['news_text']
    
    # Check if the input is too short
    if len(text.split()) < 10:
        return render_template('index.html', prediction_text="This sentence is too short. Please provide a longer news article for accurate detection.")
    
    # Convert the text into numerical features using the vectorizer
    text_tfidf = vectorizer.transform([text])  # No additional preprocessing
    
    # Make predictions using Naïve Bayes
    nb_prediction = nb_model.predict(text_tfidf)
    
    # Map the prediction to a label
    nb_result = "Fake News" if nb_prediction[0] == 0 else "Real News"
    
    # Return the result to the user
    return render_template('index.html', prediction_text=f'Naïve Bayes Prediction: {nb_result}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
