import json
from flask import Flask
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate

app = Flask(__name__)
CORS(app)

@app.route('/api/questions', methods=['POST'])

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")  # Replace with your Weaviate instance URL

def get_similar_questions(input_question: str):
    # Tokenize the input question and convert to a vector
    inputs = tokenizer(input_question, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs)
    question_vector = embeddings.pooler_output[0].numpy()

    # Query Weaviate for similar questions
    query = {
        "vector": question_vector.tolist(),
        "n": 10
    }
    result = client.query.post(query)

    # Extract similar questions from the result
    similar_questions = []
    for item in result["data"]:
        similar_questions.append(item["question"])

    return similar_questions

if __name__ == '__main__':
    app.run(debug=True, port=5000)
