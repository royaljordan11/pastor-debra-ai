# app_min.py
from flask import Flask, jsonify

app = Flask(__name__)

# Simple health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Simple root route so hitting "/" in the browser works
@app.route("/", methods=["GET"])
def index():
    return (
        "<h1>Pastor Debra AI — Minimal</h1>"
        "<p>If you see this page, the backend is up.</p>"
        "<p>Health: <a href='/health'>/health</a></p>"
    ), 200

# ⚠️ IMPORTANT: no `if __name__ == '__main__': app.run(...)` block here.
# gunicorn will import `app_min:app` and run it itself.
