from flask import Flask, request, jsonify
from flask_cors import CORS
import transformers
import spacy
import weaviate

app = Flask(__name__)
CORS(app)
