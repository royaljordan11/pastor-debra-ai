# app_min.py
import os
from flask import Flask, jsonify

app = Flask(__name__)

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/")
def index():
    return "Pastor Debra AI is running on Railway ✅"

if __name__ == "__main__":
    # Railway sets PORT as an env variable
    port = int(os.environ.get("PORT", 8080))
    # Bind to 0.0.0.0 so Railway can reach it
    app.run(host="0.0.0.0", port=port)
