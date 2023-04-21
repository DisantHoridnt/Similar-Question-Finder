from flask import Flask, request, jsonify
from flask_cors import CORS
import transformers
import spacy
import weaviate

app = Flask(__name__)
CORS(app)
@app.route('/api/questions', methods=['POST'])
def get_similar_questions():
    # Your code for processing the question, NLP, and vector similarity search will go here
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
