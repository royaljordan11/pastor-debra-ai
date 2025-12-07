# app_min.py
from flask import Flask, jsonify

app = Flask(__name__)

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/")
def index():
    return "Pastor Debra AI is running on Railway ✅"
