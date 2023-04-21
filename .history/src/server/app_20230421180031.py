# import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate

app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

@app.route('/api/questions', methods=['POST'])

def get_similar_questions():
    input_question = request.json.get("input_question", "")
    # Tokenize the input question and convert to a vector
    inputs = tokenizer(input_question, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs)
    question_vector = embeddings.pooler_output[0].numpy()

    auth_config = weaviate.auth.AuthApiKey(api_key="<YOUR-WEAVIATE-API-KEY>")  # Replace w/ your API Key for the Weaviate instance

# Instantiate the client with the auth config
client = weaviate.Client(
    url="https://some-endpoint.weaviate.network",  # Replace w/ your endpoint
    auth_client_secret=auth_config
)
    # Query Weaviate for similar questions
    weaviate_results = client.query.get(
        "Question", ["question"]
    ).with_near_vector(
        question_vector.tolist()
    ).with_limit(10).do()

    # Extract similar questions from the result
    similar_questions = [item["question"] for item in weaviate_results["data"]["Get"]["Question"]]

    return jsonify(similar_questions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)