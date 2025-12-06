# app_min.py
import os
from flask import Flask, jsonify

app = Flask(__name__)

# Simple health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Simple root route so hitting "/" in the browser also works
@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>Pastor Debra AI — Minimal</h1>"
        "<p>If you see this page, the backend is up.</p>"
        "<p>Health: <a href='/health'>/health</a></p>"
    ), 200

if __name__ == "__main__":
    # ✅ IMPORTANT: bind to Railway's PORT env, fallback to 8080 locally
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
