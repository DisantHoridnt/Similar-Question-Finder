import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
from weaviate import WeaviateClient

app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")  # Replace with your Weaviate instance URL

@app.route('/api/questions', methods=['POST'])
def get_similar_questions():
    input_question = request.json.get("input_question", "")
    # Tokenize the input question and convert to a vector
    inputs = tokenizer(input_question, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs)
    question_vector = embeddings.pooler_output[0].numpy()

    # Create a Weaviate client
    client = WeaviateClient("http://localhost:8080")

    # Query Weaviate for similar questions
    query = {
        "className": "Question",  # Assuming your class name is 'Question'
        "filter": {
            "nearVector": {
                "vector": question_vector.tolist(),
                "maxDistance": 1.0  # You can adjust this value as needed
            }
        },
        "properties": ["question"],
        "limit": 10
    }
    result = client.get(query)

    # Extract similar questions from the result
    similar_questions = []
    for item in result["data"]["objects"]:
        similar_questions.append(item["question"])

    return jsonify(similar_questions)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
