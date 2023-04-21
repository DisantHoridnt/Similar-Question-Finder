import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate

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
    client = weaviate.Client("http://localhost:8080")

    # Define the GraphQL query for similar questions
    graphql_query = f"""
    {{
        Get {{
            Question(nearVector: {question_vector.tolist()}, maxDistance: 1.0, limit: 10) {{
                question
            }}
        }}
    }}
    """

    # Query Weaviate for similar questions
    result = client.query(graphql_query)

    # Extract similar questions from the result
    similar_questions = [item["question"] for item in result["data"]["Get"]["Question"]]

    return jsonify(similar_questions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
