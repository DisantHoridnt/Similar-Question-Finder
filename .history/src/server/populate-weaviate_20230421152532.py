import weaviate

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")

# Define the schema
schema = {
  "classes": [
    {
      "class": "Question",
      "properties": [
        {
          "dataType": ["text"],
          "name": "question"
        },
        {
          "dataType": ["float"],
          "name": "vector",
          "vectorIndexType": "hnsw",
          "vectorIndexConfig": {
            "efConstruction": 128,
            "m": 64
          }
        }
      ]
    }
  ]
}

# Create the schema in Weaviate
client.schema.create(schema)

# Index some questions and their vector representations
questions = [
  {"question": "What is the capital of France?", "vector": [0.1, 0.2, 0.3, ...]},
  {"question": "How to learn Python?", "vector": [0.4, 0.5, 0.6, ...]}
]

for question in questions:
    client.data_object.create(question, "Question")
