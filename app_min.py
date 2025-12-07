# app_min.py
from flask import Flask, jsonify
from flask import render_template


app = Flask(__name__)

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

from flask import render_template

@app.route("/")
def index():
    return render_template("Pastor.html")
