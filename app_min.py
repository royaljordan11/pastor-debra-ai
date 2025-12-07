# app_min.py
from flask import Flask, jsonify

app = Flask(__name__)

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Root route
@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>Pastor Debra AI — Minimal</h1>"
        "<p>If you see this page, the backend is up.</p>"
        "<p>Health: <a href=\"/health\">/health</a></p>"
    ), 200
