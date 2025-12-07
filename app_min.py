
"""
Pastor Debra Chatbot Backend (Flask) — Hybrid (T5/ONNX + GPT)
Hardened: .env loading, videos.json auto-stub, T5 tokenizer sanity checks + fast→slow fallback.

Endpoints
- GET  /              -> serves Pastor.html
- GET  /mom.mp4       -> serves intro video (mp4 preferred)
- GET  /mom.mov       -> serves intro video (mov fallback)
- GET  /videos        -> videos list (mom.mp4/mom.mov injected first if present)
- GET  /health        -> model/docs status (incl. GPT availability, budgets, ONNX providers)
- POST /reload        -> hot reload corpora
- GET  /search?q=...  -> debug blended retrieval
- GET|POST /destiny_theme[?dob|?name]
- POST /chat          -> main chat (router: T5 or GPT or forced)
"""

import os, re, json, logging, time, hashlib, threading, datetime
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from flask import Flask, request, jsonify, session, Response
import hashlib
import time
from datetime import datetime, timezone
import random



from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer
import onnxruntime as ort

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
