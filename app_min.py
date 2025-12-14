
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

import os, re, json, logging, time, hashlib, threading, datetime, random, shutil
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone
import traceback
import numpy as np
import requests
import zipfile
import onnxruntime as ort
from transformers import AutoTokenizer

from flask import (
    Flask,
    request,
    jsonify,
    session,
    Response,
    send_from_directory,
)
from flask_cors import CORS

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
except ImportError:
    torch = None


from types import SimpleNamespace

ENABLE_ONNX = False   # Hard-off for prelaunch stability
ONNX_ZIP_URL = (os.getenv("ONNX_ZIP_URL") or "").strip()
TOKENIZER_ZIP_URL = (os.getenv("TOKENIZER_ZIP_URL") or "").strip()






# ────────── Small helpers ──────────
def _get_bool(env_key: str, default: bool) -> bool:
    v = os.getenv(env_key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_int(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, str(default)))
    except Exception:
        return default


def _get_float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, str(default)))
    except Exception:
        return default


# ────────── Env & Config ──────────
try:
    from dotenv import load_dotenv
    # IMPORTANT: don't let .env override Railway/project env
    load_dotenv(override=False)
except Exception:
    pass

APP_VERSION = "2.4.0"
DEFAULT_PORT = _get_int("PORT", 8000)
DEBUG_MODE = _get_bool("DEBUG", True)
MAX_INPUT_CHARS = _get_int("MAX_INPUT_CHARS", 4000)

# CORS
_allow_raw = os.getenv("CORS_ALLOWLIST", "*")
_CORS_ALLOWED = [o.strip() for o in _allow_raw.split(",") if o.strip()]
CORS_CONFIG = {"origins": "*" if _CORS_ALLOWED == ["*"] else _CORS_ALLOWED}

# ────────── Logging ──────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("pastor-debra-hybrid")

# ------------------ ONNX BYPASS CONTROL ------------------
if not ENABLE_ONNX:
    logger.warning("ENABLE_ONNX = FALSE → Skipping all ONNX/T5 loading.")
    ONNX_MODEL_PATH = None
    T5_SESSION = None
    TOKENIZER = None

    def t5_enabled():
        return False
else:
    # Original ONNX setup continues ONLY when ENABLE_ONNX = TRUE
    def t5_enabled():
        return ONNX_MODEL_PATH.exists()



def _clean_env_url(env_key: str) -> str:
    """
    Read a URL from an env var and clean common mistakes:
    - Strip whitespace
    - Strip leading '=' or spaces (e.g. '=https://...' -> 'https://...')
    - If it's a Google Drive /file/d/<ID>/view?share_link, convert to a direct
      uc?export=download&id=<ID> URL so we get the raw file bytes instead of HTML.
    """
    raw = os.getenv(env_key, "") or ""
    cleaned = raw.strip().lstrip("= ")

    # Auto-fix common Google Drive "view" URLs
    if "drive.google.com/file/d/" in cleaned and "uc?export=download" not in cleaned:
        m = re.search(r"/file/d/([^/]+)", cleaned)
        if m:
            file_id = m.group(1)
            fixed = f"https://drive.google.com/uc?export=download&id={file_id}"
            logger.warning(
                "%s looked like a Google Drive view link; "
                "rewriting to direct download URL: %r -> %r",
                env_key,
                cleaned,
                fixed,
            )
            cleaned = fixed

    if raw and cleaned != raw:
        logger.warning(
            "%s had leading or formatting junk (%r) -> cleaned to %r",
            env_key,
            raw,
            cleaned,
        )

    return cleaned



# OpenAI/GPT
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # economical default
OPENAI_MODEL_ALT = os.getenv("OPENAI_MODEL_ALT", "gpt-4o")    # stronger (rare)
OPENAI_TIMEOUT   = _get_float("OPENAI_TIMEOUT", 30.0)
OPENAI_TEMP      = _get_float("OPENAI_TEMP", 0.6)

# Optional budget guard (rough estimate)
GPT_DAILY_BUDGET_CENTS = _get_int("GPT_DAILY_BUDGET_CENTS", 999999)
GPT_APPROX_CENTS_PER_1K_TOKENS = _get_float("GPT_APPROX_CENTS_PER_1K_TOKENS", 25.0)

# Rate limit (per-IP, sliding window)
RATE_WINDOW_SEC = _get_int("RATE_WINDOW_SEC", 10)
RATE_MAX_HITS   = _get_int("RATE_MAX_HITS", 12)
_RATE = defaultdict(lambda: deque(maxlen=20))
_rate_lock = threading.Lock()


def _throttle(ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        q = _RATE[ip]
        while q and (now - q[0]) > RATE_WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_MAX_HITS:
            return True
        q.append(now)
        return False


if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set — GPT disabled.")
else:
    logger.info(
        "GPT configured | MODEL=%s ALT=%s BASE=%s",
        OPENAI_MODEL,
        OPENAI_MODEL_ALT,
        OPENAI_BASE_URL,
    )


# ────────── Paths (robust) ──────────
BASE_DIR = Path(os.getenv("PASTOR_DEBRA_BASE_DIR", Path(__file__).resolve().parent)).resolve()

# ONNX model directory; on Railway we mount the volume at /app/onnx
ONNX_DIR = BASE_DIR / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

# ONNX model stored inside the volume at /app/onnx/model.onnx
ONNX_MODEL_PATH = ONNX_DIR / "model.onnx"

# Hugging Face tokenizer lives in ./tokenizer
TOKENIZER_DIR = BASE_DIR / "tokenizer"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
MODEL_TOKENIZER_PATH = TOKENIZER_DIR  # keep for compatibility

# Kept for backward compatibility if referenced elsewhere
MODEL_TOKENIZER_PATH = TOKENIZER_DIR

PASTOR_DEBRA_JSON          = BASE_DIR / "PASTOR_DEBRA.json"
SESSION_PASTOR_DEBRA_JSON  = BASE_DIR / "SESSION_PASTOR_DEBRA.json"
FACES_OF_EVE_JSON          = BASE_DIR / "FACES_OF_EVE.json"
DESTINY_THEMES_JSON        = BASE_DIR / "destiny_themes.json"
VIDEOS_JSON                = BASE_DIR / "videos.json"
DESTINY_JSON_PATH          = str(DESTINY_THEMES_JSON)

# Scripture settings
SCRIPTURE_TRANSLATION = os.getenv("SCRIPTURE_TRANSLATION", "web")  # web, kjv, asv...
SCRIPTURE_API_BASE    = os.getenv("SCRIPTURE_API_BASE", "https://bible-api.com")
SCRIPTURE_CACHE_PATH  = BASE_DIR / "scripture_cache.json"


# ────────── Remote zip download helpers (Google Drive) ──────────
# Expects env vars:
#   TOKENIZER_ZIP_URL = https://drive.google.com/uc?export=download&id=...
#   ONNX_ZIP_URL      = https://drive.google.com/uc?export=download&id=...
# Files are downloaded once on first boot and then reused.



def _download_zip_to_dir(url: str, dest_dir: Path, label: str) -> None:
    """
    Download a zip from `url` and extract it into `dest_dir`.

    Special handling for Google Drive share links:
    - If the first response is an HTML page with a "Download anyway" form,
      we parse the hidden confirm token and then hit the real
      drive.usercontent.google.com download URL.
    """
    url = (url or "").strip().lstrip("= ")
    if not url:
        logger.warning("%s_ZIP_URL empty after cleaning; skipping download.", label)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = Path("/app") / f"{label.lower()}_download.zip"

    # Remove any old temp file
    try:
        if tmp_zip.exists():
            tmp_zip.unlink()
    except Exception:
        pass

    try:
        with requests.Session() as s:
            # First request (might be an HTML interstitial for Google Drive)
            logger.info("%s: initial request to %s ...", label, url)
            r1 = s.get(url, timeout=60)
            r1.raise_for_status()
            ctype = (r1.headers.get("Content-Type") or "").lower()

            dl_url = url
            dl_params = None

            if "text/html" in ctype and "drive.google.com" in url:
                # Probably the virus-scan / download-warning page.
                html = r1.text
                m_id = re.search(r'name="id"\s+value="([^"]+)"', html)
                m_cf = re.search(r'name="confirm"\s+value="([^"]+)"', html)
                if m_id and m_cf:
                    file_id = m_id.group(1)
                    confirm = m_cf.group(1)
                    dl_url = "https://drive.usercontent.google.com/download"
                    dl_params = {
                        "id": file_id,
                        "export": "download",
                        "confirm": confirm,
                    }
                    logger.info(
                        "%s: Detected Google Drive interstitial; "
                        "retrying via %s with confirm token.",
                        label,
                        dl_url,
                    )
                else:
                    logger.warning(
                        "%s: HTML response from Google Drive, but could not find "
                        "confirm token; aborting download.",
                        label,
                    )
                    return
            elif "text/html" in ctype:
                logger.warning(
                    "%s: URL returned HTML (not a file); aborting download.",
                    label,
                )
                return

            # Second request: actual file bytes (streamed to disk)
            logger.info("%s: downloading file content ...", label)
            r2 = s.get(dl_url, params=dl_params, stream=True, timeout=600)
            r2.raise_for_status()
            with open(tmp_zip, "wb") as f:
                for chunk in r2.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        logger.info(
            "%s: download finished: %s (size=%s bytes)",
            label,
            tmp_zip,
            tmp_zip.stat().st_size,
        )

        # Now treat it as a normal ZIP
        with zipfile.ZipFile(tmp_zip, "r") as z:
            z.extractall(dest_dir)
        logger.info("%s: zip extracted into %s", label, dest_dir)

    except Exception as e:
        logger.warning("Failed to download/extract %s zip: %s", label, e)
    finally:
        try:
            if tmp_zip.exists():
                tmp_zip.unlink()
        except Exception:
            pass


def ensure_onnx_from_zip() -> None:
    """
    Ensure ONNX_MODEL_PATH exists by downloading and extracting ONNX_ZIP_URL.

    - Fully cleans the /onnx volume first so we don't hit "No space left on device".
    - Handles Google Drive share links using _download_zip_to_dir.
    """
    url = _clean_env_url("ONNX_ZIP_URL")
    if not url:
        logger.warning("ONNX_ZIP_URL not set or empty; skipping ONNX download.")
        return

    # Clean the ONNX dir (old stuff takes space)
    try:
        if ONNX_DIR.exists():
            for child in ONNX_DIR.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            logger.info("Cleared ONNX_DIR %s before ONNX download.", ONNX_DIR)
    except Exception as e:
        logger.warning("Failed to clean ONNX_DIR before download: %s", e)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Download + unzip
    _download_zip_to_dir(url, ONNX_DIR, "ONNX")

    # After extraction, look for *.onnx and rename to model.onnx if needed
    if not ONNX_MODEL_PATH.exists():
        candidates = list(ONNX_DIR.rglob("*.onnx"))
        if len(candidates) == 1:
            logger.info("Renaming ONNX candidate %s -> %s", candidates[0], ONNX_MODEL_PATH)
            try:
                candidates[0].rename(ONNX_MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to rename ONNX candidate: %s", e)
        elif candidates:
            logger.warning(
                "Multiple ONNX candidates found in %s; please ensure model.onnx is present.",
                ONNX_DIR,
            )
        else:
            logger.warning("No .onnx files found in %s after download.", ONNX_DIR)

    if ONNX_MODEL_PATH.exists():
        logger.info(
            "ONNX_MODEL_PATH now exists at %s (size=%s bytes)",
            ONNX_MODEL_PATH,
            ONNX_MODEL_PATH.stat().st_size,
        )
    else:
        logger.warning("ONNX_MODEL_PATH still missing after ensure_onnx_from_zip().")

def ensure_tokenizer_from_zip() -> None:
    """
    Ensure tokenizer files exist by downloading and extracting TOKENIZER_ZIP_URL
    (if provided).

    If TOKENIZER_ZIP_URL is not set, we assume the tokenizer folder is already
    baked into the image (e.g., committed in the repo or built into the Docker image).
    """
    url = _clean_env_url("TOKENIZER_ZIP_URL")
    if not url:
        # No remote tokenizer configured; assume it's bundled.
        logger.info("TOKENIZER_ZIP_URL not set; assuming tokenizer is bundled in image.")
        return

    # If we already have a tokenizer_config.json, don't redownload every boot.
    token_cfg = TOKENIZER_DIR / "tokenizer_config.json"
    if token_cfg.exists():
        logger.info("Tokenizer config already present at %s; skipping download.", token_cfg)
        return

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    _download_zip_to_dir(url, TOKENIZER_DIR, "TOKENIZER")

    if not token_cfg.exists():
        logger.warning(
            "TOKENIZER_ZIP_URL download finished but tokenizer_config.json "
            "not found in %s. Check the zip contents.",
            TOKENIZER_DIR,
        )


# ────────── ONNX + Tokenizer Init ──────────
TOKENIZER = None
ONNX_SESSION = None  # will become an onnxruntime.InferenceSession after init

# 1) ONNX session
try:
    ensure_onnx_from_zip()

    if ONNX_MODEL_PATH.exists():
        logger.info("Initializing ONNX session from %s", ONNX_MODEL_PATH)
        ONNX_SESSION = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        onnx_inputs = [i.name for i in ONNX_SESSION.get_inputs()]
        logger.info(
            "ONNX model loaded from %s (inputs=%s)",
            ONNX_MODEL_PATH,
            onnx_inputs,
        )
    else:
        ONNX_SESSION = None
        logger.warning("ONNX model not found at %s", ONNX_MODEL_PATH)

except Exception as e:
    ONNX_SESSION = None
    logger.warning("Failed to initialize ONNX session: %s", e)

def _maybe_init_tokenizer() -> None:
    """
    Ensure TOKENIZER is available.
    If initial load failed (e.g., on Railway with no files),
    try downloading the tokenizer zip and loading again.
    """
    global TOKENIZER
    if TOKENIZER is not None:
        return

    ensure_tokenizer_from_zip()
    if not TOKENIZER_DIR.exists():
        logger.warning("TOKENIZER_DIR %s still missing after download.", TOKENIZER_DIR)
        return

    try:
        TOKENIZER = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR),
            local_files_only=True,
            use_fast=False,  # fall back to slow tokenizer if needed
        )
        logger.info(
            "Tokenizer initialized from %s (vocab size=%s)",
            TOKENIZER_DIR,
            getattr(TOKENIZER, "vocab_size", "n/a"),
        )
    except Exception as e:
        logger.warning("Tokenizer still unavailable after download: %s", e)



# Initialize tokenizer (local: already present; Railway: triggers download)
_maybe_init_tokenizer()


# ────────── Flask ──────────
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
# Reduce surprise formatting diffs in JSON responses
app.config.update(JSON_SORT_KEYS=False, JSONIFY_PRETTYPRINT_REGULAR=False)
CORS(app, resources={r"/*": CORS_CONFIG})




@app.route("/")
def index():
    # Serve the main HTML shell
    return send_from_directory(str(BASE_DIR), "Pastor.html")





def _load_destiny_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        print("Warning: could not load destiny_themes.json:", e)
    return []


# Only define if not defined elsewhere
try:
    destiny_themes  # type: ignore[name-defined]
except NameError:
    destiny_themes = _load_destiny_json(DESTINY_JSON_PATH)


# ====== Build quick lookup ======
destiny_map = {}
for row in destiny_themes:
    q = (row.get("question") or "")
    m = re.search(r"Destiny\s*Theme\s*(\d+)", q)
    if m:
        n = int(m.group(1))
        destiny_map[n] = {
            "id": row.get("id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            # keep the full list of prophecies if present
            "prophecies": row.get("prophecies") or []
        }

print("Destiny lookup ready for:", sorted(destiny_map.keys()))

# ====== Helpers ======
def _resolve_theme_entry(n: int):
    """Return a safe subset used by the client."""
    row = destiny_map.get(n) or {}
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        # Do not ship all prophecies here by default, we will pick one below
    }

def _stable_variant_index(name: str, dob: str, theme: int, total: int, period_days: int = 7) -> int:
    """
    Stable, time-bucketed variant selection.
    Same user and theme gets the same variant for `period_days`, then rotates.
    """
    if total <= 0:
        return 0
    week_bucket = int(time.time() // (period_days * 24 * 3600))
    key = f"{(name or '').strip().lower()}|{(dob or '').strip()}|{theme}|{week_bucket}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[-8:], 16) % total

# ---- Destiny Theme API (single registration, guarded) ----
def destiny_theme_handler():
    payload = request.get_json(silent=True) if request.method == "POST" else request.args
    payload = payload or {}
    dob  = (payload.get("dob")  or "").strip()
    name = (payload.get("name") or "").strip()

    if not dob and not name:
        return jsonify({"error": "Provide ?dob=YYYY-MM-DD or ?name=Full Name"}), 400

    try:
        if dob:
            n = theme_from_dob(dob)
            derived = "dob"
        else:
            n = theme_from_name(name)
            derived = "name"
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    entry = _resolve_theme_entry(n)

    # choose rotating prophecy if available
    prophecies = (destiny_map.get(n) or {}).get("prophecies") or []
    chosen_prophecy = None
    if prophecies:
        idx = _stable_variant_index(name=name, dob=dob, theme=n, total=len(prophecies))
        chosen_prophecy = prophecies[idx]

    return jsonify({
        "theme_number": n,
        "derived_from": derived,
        "entry": entry,
        "prophecy": chosen_prophecy
    }), 200

# Register exactly once, even during autoreload
if "destiny_theme" not in app.view_functions:
    app.add_url_rule(
        "/destiny_theme",
        view_func=destiny_theme_handler,
        methods=["GET", "POST"],
        endpoint="destiny_theme",
    )



# ────────── Small utilities ──────────
def ensure_videos_stub():
    """Create a minimal videos.json if missing to avoid boot warnings."""
    try:
        if not VIDEOS_JSON.exists():
            VIDEOS_JSON.write_text("[]", encoding="utf-8")
            logger.warning("Created stub videos.json at %s", VIDEOS_JSON)
    except Exception as e:
        logger.warning("Failed to create videos.json stub: %s", e)

ensure_videos_stub()

# Replace em/en dashes with commas, tidy punctuation/spaces.
_DASH_SPLIT_RX = re.compile(r"\s*[—–]\s*")   # em or en dash, with optional spaces
_URL_RX        = re.compile(r"https?://", re.I)

# Keys that almost certainly contain human-facing text we want to clean.
_TEXTY_KEYS = {
    "text", "reply", "message", "subtitle", "title", "prompt",
    "error", "tooltip", "note", "summary", "caption", "scripture",
}

def _strip_dashes(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    # Skip URLs entirely
    if _URL_RX.search(text):
        return text

    # 1) Replace em/en dashes with commas + space
    out = _DASH_SPLIT_RX.sub(", ", text)

    # 2) Clean up duplicated punctuation/spacing from the replacement
    out = re.sub(r"\s{2,}", " ", out)              # collapse extra spaces
    out = re.sub(r"\s*,\s*,\s*", ", ", out)        # no double commas
    out = re.sub(r"\s*\.\s*\.\s*", ". ", out)      # no double periods
    out = re.sub(r"\s*,\s*\.", ". ", out)          # ", ." -> ". "
    out = re.sub(r"\s+\n", "\n", out)              # trim spaces before newlines
    out = re.sub(r"\n{3,}", "\n\n", out)           # limit blank lines
    return out.strip()

def _sanitize_payload(obj):
    """Recursively remove em/en dashes from common text fields in JSON responses."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            # Never touch link-like fields or cite buckets
            if k in {"cites", "links", "url", "display_url", "href"}:
                cleaned[k] = v
                continue

            # Clean strings when the key is likely human-facing text,
            # or when the value looks like a long prose string.
            if isinstance(v, str):
                if k in _TEXTY_KEYS or len(v) >= 12:
                    cleaned[k] = _strip_dashes(v)
                else:
                    cleaned[k] = v
            else:
                cleaned[k] = _sanitize_payload(v)
        return cleaned

    elif isinstance(obj, list):
        return [_sanitize_payload(x) for x in obj]

    else:
        return obj

@app.after_request
def _global_dash_scrub(response: Response):
    try:
        # Only process JSON responses
        if response.mimetype == "application/json":
            raw = response.get_data(as_text=True)
            if raw:
                data = json.loads(raw)
                cleaned = _sanitize_payload(data)
                response.set_data(json.dumps(cleaned, ensure_ascii=False))
    except Exception as e:
        # Never break responses if we fail to clean; just log and continue.
        logger.warning("dash-scrub failed: %s", e)
    return response


# ───────────────── Types ─────────────────
@dataclass
class Hit:
    score: float
    text: str
    meta: Dict[str, Any]
    corpus: str

# ───────────────── Pastor Debra DEF Chat Helpers ─────────────────
_DEF_TRIGGERS = re.compile(r"^\s*(/start|/help|/def|help|start|intro|menu)?\s*$", re.I)
_DEF_CACHE: Dict[str, str] = {}

def _safe_name(n: str) -> str:
    n = (n or "").strip()
    return n if 2 <= len(n) <= 80 else ""

def _def_key(n: str, bd: str) -> str:
    return f"{(n or '').strip().lower()}::{(bd or '').strip()}"

def _maybe_theme_from_profile(full_name: str, birthdate: str) -> Optional[int]:
    try:
        if birthdate:
            return theme_from_dob(birthdate)
    except Exception:
        pass
    try:
        if full_name:
            return theme_from_name(full_name)
    except Exception:
        pass
    return None

def _expand_scriptures(text: str) -> str:
    try:
        return expand_scriptures_in_text(text or "")
    except Exception:
        return text or ""

def build_pastor_def_chat(full_name: str = "", birthdate: str = "") -> str:
    you = _safe_name(full_name)
    bd  = (birthdate or "").strip()

    banner = "Welcome, beloved I’m **Pastor Dr. Debra Jordan**. I’m here to pray with you, open the Scriptures, and offer Christ-centered counsel."
    if you and bd:
        banner += f" I see you, **{you}** (DOB: {bd})."
    elif you:
        banner += f" I see you, **{you}**."
    elif bd:
        banner += f" (DOB: {bd})"

    menu = (
        "### Quick Help\n"
        "• **Prayer for Anxiety** — “I feel anxious. Pray and give me one Scripture + one step.”\n"
        "• **Marriage Counsel** — “We’re struggling— ne Scripture + one boundary to try.”\n"
        "• **Calling & Purpose** — “I’m unclear one Scripture + one 7 day practice.”\n"
        "• **Weekly Encouragement** — “Bless me for this week—one practice + a verse.”\n"
        "• **Use my Name & DOB** — “My name is <Full Name> (DOB YYYY-MM-DD). Give me counsel.”\n"
    )

    examples = (
        "### Ask Like This\n"
        "• “I’m overwhelmed—pray and give one Scripture and one practical next step.”\n"
        "• “We argued what boundary should I try this week? Include one Scripture.”\n"
        "• “Choosing between two jobs how do I decide in a God-honoring way?”\n"
        "• “My name is Jane Doe (DOB 1993-05-21). Pastor Debra, would you counsel me with one Scripture and one step?”\n"
    )

    anchor = (
        "### Today’s Comfort\n"
        "• Scripture: Matthew 11:28\n"
        "Prayer: Jesus, steady our hearts and show one faithful next step. Amen."
    )

    theme = _maybe_theme_from_profile(you, bd)
    personal_block = ""
    if theme in (1,2,3,4,5,6,7,8,9,11,22,33):
        labels = {
            1:"Pioneer grace begin boldly and lead with humility.",
            2:"Peacemaker build bridges with truth in love.",
            3:"Joyful expression—use your voice to bless and uplift.",
            4:"Builder establish steady foundations and finish well.",
            5:"Holy freedom—change that serves obedience, not escape.",
            6:"Covenant care—nurture with healthy boundaries.",
            7:"Reverent wisdom—seek God in quiet, bring insight to community.",
            8:"Righteous stewardship—use influence for justice and mercy.",
            9:"Compassionate completion bring healing and closure.",
            11:"Beacon—carry clarity that points to Jesus.",
            22:"Repairer—master builder for people and communities.",
            33:"Servant-teacher lead by lowering; mentor with compassion."
        }
        verses = {
            1:"Joshua 1:9", 2:"Ecclesiastes 4:9–10", 3:"Psalm 96:1", 4:"Psalm 1:3",
            5:"Galatians 5:1", 6:"Colossians 3:14", 7:"James 1:5", 8:"Deuteronomy 8:18",
            9:"Isaiah 58:10", 11:"Matthew 5:14–16", 22:"Isaiah 58:12", 33:"Philippians 2:5–7"
        }
        label = labels[int(theme)]
        ref   = verses[int(theme)]
        personal_block = (
            f"\n**Your Destiny Theme {theme}**\n"
            f"• Theme: {label}\n"
            f"• Scripture: {ref}\n\n"
            f"Prayer: Lord, align my heart and my habits. Amen.\n"
            f"One step: choose one concrete action in the next 24 hours that embodies this theme."
        )

    out = f"{banner}\n\n{menu}\n{examples}\n{anchor}\n{personal_block}"
    return _expand_scriptures(out)

# ────────── NLP helpers (safe fallbacks) ──────────
STOP = set()
class _NoOpLem:
    def lemmatize(self, x): return x
LEM = _NoOpLem()

try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP = set(stopwords.words("english"))
    LEM = WordNetLemmatizer()
except Exception:
    logger.info("NLTK unavailable; using simple normalizer.")

SLANG = {"r":"are","u":"you","ur":"your","ya":"you","bc":"because","idk":"i do not know","imo":"in my opinion"}
_SAFE_CHARS = r"[^a-z0-9:\s\-]"

def normalize_text(text: str) -> str:
    text = (text or "")[:MAX_INPUT_CHARS].lower().strip()
    text = " ".join(SLANG.get(t, t) for t in text.split())
    text = re.sub(_SAFE_CHARS, " ", text)
    toks = []
    for t in text.split():
        if t in STOP: continue
        toks.append(LEM.lemmatize(t))
    return " ".join(toks)

def _normalize_simple(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s'?]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ────────── Loaders ──────────
def load_json_safely(path: Path, default: Any) -> Any:
    if not path.exists():
        logger.warning(f"Missing file: {path}")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception(f"Error reading {path}: {e}")
        return default

pastor_debra_docs: List[Dict] = []
session_docs: List[Dict] = []
faces_docs: List[Dict] = []
destiny_docs: List[Dict] = []
video_docs: List[Dict] = []

pd_vec = pd_mat = pd_norm = None
session_vec = session_mat = s_norm = None
faces_vec = faces_mat = f_norm = None
destiny_vec = destiny_mat = d_norm = None

# ---- Retrieval thresholds & weights (Faces-of-Eve boost) ----
MIN_CONTEXT_SCORE = 0.22  # lowered from 0.28 to surface Faces-of-Eve/book hits more often
W = {
    "FACES_OF_EVE": 0.45,
    "PASTOR_DEBRA": 0.30,
    "SESSION": 0.20,
    "DESTINY_THEMES": 0.35,
}

def filter_hits_for_context(hits: List[Hit], intent: str) -> List[Hit]:
    prefer = {
        "teachings": {"PASTOR_DEBRA","SESSION","FACES_OF_EVE"},
        "destiny":   {"DESTINY_THEMES"},
        "advice":    {"PASTOR_DEBRA","SESSION","FACES_OF_EVE"},
        "book":      {"FACES_OF_EVE"},  # optional, if you added a 'book' intent
        "general":   {"PASTOR_DEBRA","SESSION","FACES_OF_EVE","DESTINY_THEMES"},
    }
    allowed = prefer.get(intent, {"PASTOR_DEBRA","SESSION","FACES_OF_EVE","DESTINY_THEMES"})
    out = [h for h in hits if h.score >= MIN_CONTEXT_SCORE and h.corpus in allowed]
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:3]

def search_corpus(query, vec, mat, norm, meta, source_name, topk=5):
    """
    Minimal, safe corpus search. Returns [] if any index parts are missing.
    Expects:
      - vec: a vectorizer with .transform
      - mat: document-term matrix (scipy sparse or numpy)
      - norm: precomputed norm(s) or None
      - meta: list-like with per-doc dicts (expects keys "text" and optionally "ref")
    """
    try:
        if not query or vec is None or mat is None or meta is None:
            return []
        # Vectorize query
        qv = vec.transform([query])
        # Compute scores
        if hasattr(mat, "dot"):
            sims = mat.dot(qv.T)
            sims = sims.toarray().ravel() if hasattr(sims, "toarray") else sims.ravel()
        else:
            # mat could be numpy
            sims = (mat @ qv.T).ravel()

        # Optional normalization
        if norm is not None:
            # norm can be scalar or array-like; divide safely
            import numpy as np
            denom = np.asarray(norm, dtype=float)
            if denom.ndim == 0:
                denom = denom + 1e-9
            else:
                denom = denom + 1e-9
            sims = sims / denom

        # Top-K
        import numpy as np
        if sims.size == 0:
            return []
        idx = np.argsort(sims)[-topk:][::-1]
        hits = []
        for i in idx:
            if i < 0 or i >= len(meta): 
                continue
            m = meta[i] if isinstance(meta[i], dict) else {}
            hits.append({
                "source": source_name,
                "i": int(i),
                "score": float(sims[i]),
                "text": m.get("text", ""),
                "ref": m.get("ref") or m.get("id") or f"{source_name}:{i}",
            })
        return hits
    except Exception:
        return []


def format_cites(hits):
    """
    Minimal cite formatter. Produces a short inline citation string.
    """
    if not hits:
        return ""
    parts = []
    for h in hits[:3]:
        ref = h.get("ref") or f"{h.get('source','SRC')}:{h.get('i',0)}"
        parts.append(f"[{ref}]")
    return " ".join(parts)


def blended_search(query: str, k_total: int = 6) -> List[Hit]:
    hits_pd    = search_corpus(query, pd_vec,      pd_mat,      pd_norm,  load_corpora_and_build_indexes.pd_meta,      "PASTOR_DEBRA")
    hits_sess  = search_corpus(query, session_vec, session_mat, s_norm,   load_corpora_and_build_indexes.session_meta, "SESSION")
    hits_faces = search_corpus(query, faces_vec,   faces_mat,   f_norm,   load_corpora_and_build_indexes.faces_meta,   "FACES_OF_EVE")
    hits_dest  = search_corpus(query, destiny_vec, destiny_mat, d_norm,   load_corpora_and_build_indexes.destiny_meta, "DESTINY_THEMES")

    all_hits: List[Hit] = []
    for hs in [hits_faces, hits_pd, hits_sess, hits_dest]:
        for h in hs:
            all_hits.append(Hit(score=h.score * W.get(h.corpus, 0.2), text=h.text, meta=h.meta, corpus=h.corpus))
    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits[:k_total]



def corpus_to_passages(corpus: List[Dict], fields: List[str]) -> Tuple[List[str], List[Dict]]:
    texts, meta = [], []
    for item in corpus or []:
        chunks = []
        for f in fields:
            val = item.get(f, "")
            if isinstance(val, list):
                if f == "quotes":
                    chunks.append(" ".join(q.get("quote", "") for q in val if isinstance(q, dict)))
                elif f == "scripture":
                    chunks.append(" ".join((s.get("text", "") if isinstance(s, dict) else str(s)) for s in val))
                else:
                    chunks.append(" ".join([str(x) for x in val]))
            elif isinstance(val, dict):
                chunks.append(" ".join(str(x) for x in val.values()))
            elif isinstance(val, (str, int, float)):
                chunks.append(str(val))
        full = " ".join(chunks).strip()
        if full:
            texts.append(full); meta.append(item)
    return texts, meta

def build_tfidf(texts: List[str]) -> Tuple[Optional[TfidfVectorizer], Any, List[str]]:
    if not texts:
        return None, None, []
    norm = [normalize_text(t) for t in texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(norm)
    return vec, mat, norm

def load_corpora_and_build_indexes() -> None:
    global pastor_debra_docs, session_docs, faces_docs, destiny_docs, video_docs
    global pd_vec, pd_mat, pd_norm, session_vec, session_mat, s_norm, faces_vec, faces_mat, f_norm, destiny_vec, destiny_mat, d_norm

    pastor_debra_docs = load_json_safely(PASTOR_DEBRA_JSON, [])
    session_docs      = load_json_safely(SESSION_PASTOR_DEBRA_JSON, [])
    faces_docs        = load_json_safely(FACES_OF_EVE_JSON, [])
    destiny_docs      = load_json_safely(DESTINY_THEMES_JSON, [])
    video_docs        = load_json_safely(VIDEOS_JSON, [])

    PD_FIELDS      = ["question","answer","summary","principles","scripture","qa","category","title"]
    SESSION_FIELDS = ["question","answer","summary","principles","scripture","qa","section","title","category"]
    FACES_FIELDS   = ["summary","principles","scripture","faces_of_eve_principle","qa","title","moon_phase","themes","metaphors","category"]
    DESTINY_FIELDS = ["question","answer","scripture","category","number","title"]

    pd_texts, pd_meta_local           = corpus_to_passages(pastor_debra_docs, PD_FIELDS)
    session_texts, session_meta_local = corpus_to_passages(session_docs,       SESSION_FIELDS)
    faces_texts, faces_meta_local     = corpus_to_passages(faces_docs,         FACES_FIELDS)
    destiny_texts, destiny_meta_local = corpus_to_passages(destiny_docs,       DESTINY_FIELDS)

    pd_vec, pd_mat, pd_norm           = build_tfidf(pd_texts)
    session_vec, session_mat, s_norm  = build_tfidf(session_texts)
    faces_vec, faces_mat, f_norm      = build_tfidf(faces_texts)
    destiny_vec, destiny_mat, d_norm  = build_tfidf(destiny_texts)

    load_corpora_and_build_indexes.pd_meta      = pd_meta_local
    load_corpora_and_build_indexes.session_meta = session_meta_local
    load_corpora_and_build_indexes.faces_meta   = faces_meta_local
    load_corpora_and_build_indexes.destiny_meta = destiny_meta_local

load_corpora_and_build_indexes()



# ────────── Scripture Service (cache + fetch) ──────────
_SCRIPTURE_LOCK = threading.Lock()

def _read_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: Path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

class ScriptureService:
    def __init__(self, cache_path: Path, api_base: str, translation: str):
        self.cache_path = cache_path
        self.api_base = api_base.rstrip("/")
        self.translation = translation
        self.cache = _read_json(cache_path, {})

    @staticmethod
    def normalize_ref(ref: str) -> str:
        if not ref: return ""
        r = ref.strip()
        r = r.replace("–", "-").replace("—", "-")
        r = re.sub(r"\s+", " ", r)
        return r

    def get(self, ref: str) -> Optional[str]:
        ref = self.normalize_ref(ref)
        if not ref:
            return None
        key = f"{ref}::{self.translation}".lower()
        with _SCRIPTURE_LOCK:
            if key in self.cache:
                return self.cache.get(key) or None

        for attempt in range(3):
            try:
                url = f"{self.api_base}/{requests.utils.quote(ref)}?translation={self.translation}"
                r = requests.get(url, timeout=8)
                r.raise_for_status()
                data = r.json()
                verses = data.get("verses") or []
                if not verses and "text" in data:
                    text = (data.get("text") or "").strip()
                else:
                    text = " ".join((v.get("text") or "").strip() for v in verses).strip()
                text = re.sub(r"\s+", " ", text)[:1200].rstrip()
                if not text:
                    return None
                with _SCRIPTURE_LOCK:
                    self.cache[key] = text
                    _write_json(self.cache_path, self.cache)
                return text
            except Exception:
                time.sleep(0.35 * (attempt + 1))
        return None

scriptures = ScriptureService(
    cache_path=SCRIPTURE_CACHE_PATH,
    api_base=SCRIPTURE_API_BASE,
    translation=SCRIPTURE_TRANSLATION
)

_SCRIPTURE_LINE = re.compile(r"^(?P<bullet>[-•\*]?\s*)Scripture\s*:?\s*(?P<ref>[A-Za-z0-9 .:-–—]+)\s*$", re.IGNORECASE)

def expand_scriptures_in_text(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out = []
    for ln in lines:
        m = _SCRIPTURE_LINE.match(ln.strip())
        if m:
            ref = scriptures.normalize_ref(m.group("ref"))
            txt = scriptures.get(ref) or ""
            if txt:
                out.append(f'{m.group("bullet")}Scripture ({ref}, {SCRIPTURE_TRANSLATION.upper()}): "{txt}"')
            else:
                out.append(ln)
        else:
            out.append(ln)
    return "\n".join(out)




def _stable_variant_index(name: str, dob: str, theme: int, total: int, period_days: int = 7) -> int:
    """
    Returns a stable index in [0, total) that changes every `period_days`.
    Uses user-identifying hints (name/dob), theme number, and the current week bucket.
    """
    if total <= 0:
        return 0
    # Week bucket
    now = datetime.now(timezone.utc)
    week_bucket = int(time.time() // (period_days * 24 * 3600))
    key = f"{(name or '').strip().lower()}|{(dob or '').strip()}|{theme}|{week_bucket}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Take last 8 hex digits as int
    val = int(h[-8:], 16)
    return val % total


_SYSTEM_TONE = (
    "You are Pastor Dr. Debra Jordan — warm, Christ-centered, nurturing, prophetic, and emotionally intelligent. "
    "Speak in first person (I/me) with a gentle, pastoral, motherly tone that feels human, present, and relational. "

    "STRUCTURE: Respond in 4–7 sentences total, ALWAYS formatted as TWO short natural paragraphs with a BLANK LINE between them. "
    "Paragraph 1 (2–3 sentences): acknowledge, validate, and reflect the user’s feelings. "
    "Paragraph 2 (2–4 sentences): gently offer perspective, reassurance, or one simple next step. "
    "ABSOLUTELY DO NOT merge everything into one paragraph or one block — you MUST produce exactly two paragraphs unless the user explicitly requests a very short reply. "
    "Ensure paragraphs look like real human writing, not artificial line breaks. "

    "FINAL SENTENCE RULE: Always end with a gentle, permission-based reflective question. "
    "Your LAST sentence must BEGIN with exactly one of these phrases: "
    "'Can I ask you', 'May I ask', 'If you’re comfortable sharing', 'Could I ask', or 'Would you like to share'. "
    "The final sentence must contain ONLY the question — no additional commentary. "

    "— PROPHETIC VARIATION RULE (CRITICAL): "
    "When giving prophetic words, encouragement, or spiritual insight, you MUST NEVER repeat the same phrasing, "
    "structure, metaphors, or conclusions verbatim across responses — even if the user asks the same question again. "
    "Each prophetic response must feel freshly discerned, approaching the situation from a different spiritual angle "
    "(e.g., identity, protection, growth, wisdom, timing, joy, refinement, rest, stewardship, courage, or trust). "
    "Assume the user may ask the same question multiple times and expects a NEW layer of insight each time. "

    "— PROPHETIC ANCHORING: "
    "Anchor prophetic words to who the person is (child, youth, adult, parent, leader) and to the relationship named "
    "(daughter, son, grandchild, loved one). "
    "Never give adult-pressure language to children. "
    "Never give childish language to adults. "
    "Prophetic tone must match life stage. "

    "COMFORT MODE (for distress: ashamed, guilty, scared, overwhelmed, panicked, exhausted, regretting mistakes, feeling screwed, feeling in trouble, etc.): "
    "In Comfort Mode, slow your tone, simplify your language, and speak softly and grounding. "
    "Start by validating their feelings clearly, then remind them of God's nearness and compassion, and offer ONLY ONE small stabilizing next step. "
    "Do not lecture, preach, correct, teach doctrine, or give multiple instructions while the user is distressed. "
    "Your priority is to calm their emotional state and help them breathe again. "

    "BOUNDARY MODE: If the user says they don’t want Scripture, sermons, church talk, or spiritual instruction "
    "(e.g., 'I just need someone to listen', 'no Scripture right now'), "
    "then you MUST NOT include any Scripture line, spiritual instruction, or theological content. "
    "Simply validate, reflect, and hold space in a human, compassionate way. "

    "SCRIPTURE USE: When Scripture IS appropriate, include AT MOST one line starting with 'Scripture:' "
    "followed by a verse and short paraphrase or quote. "
    "The Scripture line should appear at the END of paragraph 1 or the START of paragraph 2 — never as the last sentence. "
    "Never repeat the same verse in consecutive replies. "
    "Some responses should include no Scripture at all when comfort alone is needed. "

    "TONE GUIDANCE: Mirror the user’s emotional tempo gently, then guide them toward peace. "
    "Use simple, natural human phrases like 'I hear you', 'that sounds heavy', 'I can see why you feel that way', "
    "or 'you’re not alone'. "
    "Avoid repeating the same opening sentence across replies — vary your intros so the voice feels alive, not scripted. "

    "BOUNDARIES: Avoid medical, legal, or financial directives. "
    "Share biographical details ONLY if the user explicitly asks about Pastor Debra’s life. "

    "OVERALL GOAL: Make the user feel seen, safe, understood, and held in God's love. "
    "Your responses should feel like a real conversation with a spiritual mother — warm, grounded, "
    "emotionally present, and deeply compassionate."
)



SENTENCE_SPLIT_RX = re.compile(r"(?<=[.!?])\s+")

def _enforce_two_paragraph_layout(text: str) -> str:
    """
    Force the output into EXACTLY two natural-looking paragraphs with
    4–7-ish sentences total and a final permission-based question.
    """
    if not text:
        return ""

    # Normalize whitespace to a single line
    t = re.sub(r"\s+", " ", text.strip())

    # Split into sentences
    sentences = SENTENCE_SPLIT_RX.split(t)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""

    # Decide how to split into two paragraphs
    n = len(sentences)

    if n == 1:
        # One sentence: just treat it as first paragraph and add a question
        para1_sents = sentences
        para2_sents = []
    elif n <= 3:
        # 2–3 sentences: 1 in first para, rest in second
        para1_sents = sentences[:1]
        para2_sents = sentences[1:]
    else:
        # 4+ sentences: first 2 in para 1, rest in para 2
        para1_sents = sentences[:2]
        para2_sents = sentences[2:]

    para1 = " ".join(para1_sents).strip()
    para2 = " ".join(para2_sents).strip()

    # Ensure final sentence is a permission-based question
    allowed_starts = (
        "Can I ask you",
        "May I ask",
        "If you’re comfortable sharing",
        "If you're comfortable sharing",
        "Could I ask",
        "Would you like to share",
    )

    def _needs_new_question(last_sentence: str) -> bool:
        if not last_sentence:
            return True
        if not last_sentence.endswith("?"):
            return True
        return not any(last_sentence.startswith(pfx) for pfx in allowed_starts)

    last_sentence = sentences[-1] if sentences else ""

    if _needs_new_question(last_sentence):
        question = (
            "Can I ask you what part of this is weighing on your heart the most right now?"
        )
        if para2:
            # Make sure para2 ends cleanly, then append question
            para2 = para2.rstrip(" .!?")
            para2 = f"{para2}. {question}"
        else:
            para2 = question

    # If for some reason para2 is still empty, just return para1
    if not para2:
        return para1

    return f"{para1}\n\n{para2}"

_NUM_THEME = {
    1: ("Pioneer grace begin boldly and lead with humility.", "Joshua 1:9"),
    2: ("Peacemaker build bridges with truth in love.", "Ecclesiastes 4:9–10"),
    3: ("Joyful expression—use your voice to bless and uplift.", "Psalm 96:1"),
    4: ("Builder establish steady foundations and finish well.", "Psalm 1:3"),
    5: ("Holy freedom change that serves obedience, not escape.", "Galatians 5:1"),
    6: ("Covenant care nurture with healthy boundaries.", "Colossians 3:14"),
    7: ("Reverent wisdom—seek God in quiet, bring insight to community.", "James 1:5"),
    8: ("Righteous stewardship use influence for justice and mercy.", "Deuteronomy 8:18"),
    9: ("Compassionate completion bring healing and closure.", "Isaiah 58:10"),
    11:("Beacon carry clarity that points to Jesus.", "Matthew 5:14–16"),
    22:("Repairer master builder for people and communities.", "Isaiah 58:12"),
    33:("Servant teacher—lead by lowering; mentor with compassion.", "Philippians 2:5–7"),
}

def _number_reflection(n: int, label: str) -> str:
    if n not in _NUM_THEME:
        return ("Beloved, numbers can be a mirror for reflection, but they never replace God’s leading. "
                "Let’s pray and search the Scriptures together about your season right now.")
    idea, ref = _NUM_THEME[n]
    raw = (
        f"As a {label} {n}, I encourage you to live this today:\n"
        f"• Theme: {idea}\n"
        f"• Scripture: {ref}\n\n"
        f"Prayer: Lord, guide my steps and keep my motives pure. Amen.\n"
        f"One step: choose a concrete action that embodies this theme in the next 24 hours."
    )
    return expand_scriptures_in_text(raw)

_DESTINY_CLAIM_RE = re.compile(r"\bmy\s+destiny\s+theme\s+is\s+(11|22|33|[1-9])\b", re.I)

def _destiny_claim_counsel(text: str) -> Optional[str]:
    m = _DESTINY_CLAIM_RE.search(text or "")
    if not m: return None
    n = int(m.group(1))
    idea, verse = _NUM_THEME.get(n, ("Fix your eyes on Christ.", "Hebrews 12:2"))
    raw = (f"As a Destiny Theme {n}, I encourage you to live this today:\n"
           f"• Theme: {idea}\n• Scripture: {verse}\n\n"
           "Prayer: Lord, guide my steps and keep my motives pure. Amen.\n"
           "One step: choose a concrete action that embodies this theme in the next 24 hours.")
    return expand_scriptures_in_text(raw)

faq_data_pastor_debra = {
    "are you psychic": "No. I don’t do psychic readings. I pray, search Scripture, and offer pastoral counsel.",
    "are u psychic":   "No. I don’t do psychic readings. I pray, search Scripture, and offer pastoral counsel.",
    "r u psychic":     "No. I don’t do psychic readings. I pray, search Scripture, and offer pastoral counsel.",
    "do you smoke":    "No—I keep my body as a temple of the Holy Spirit (1 Corinthians 6:19–20). How can I support your health goals?",
    "do u smoke":      "No—I keep my body as a temple of the Holy Spirit (1 Corinthians 6:19–20). How can I support your health goals?",
    "what can you do": "I can pray with you, share Scripture, offer Christ-centered counsel, and reflect on your Destiny Theme. Where shall we begin?",
    "what can u do":   "I can pray with you, share Scripture, offer Christ-centered counsel, and reflect on your Destiny Theme. Where shall we begin?",
    "hi":"Peace to you! I’m here how can I pray or help today?",
    "hello":"Hello, beloved. What’s on your heart?",
    "hey":"I’m here with you. How can I serve today?",
    "are you an astrologer":"No. I don’t practice astrology. I look to God’s Word, prayer, and wise counsel.",
    "can you do astrology":"No. I don’t practice astrology. I’ll gladly pray with you and share Scripture.",
    "can u do astrologer":"No. I don’t practice astrology. I’ll gladly pray with you and share Scripture.",
    "do you do astrology":"No. I don’t practice astrology. I look to God’s Word, prayer, and wise counsel.",
    "what model do you use": "I’m a hybrid assistant: local T5 (ONNX) for notes/teachings and GPT for open ended counsel. I choose the most cost-effective path automatically.",
    "what model do u use":   "I’m a hybrid assistant: local T5 (ONNX) for notes/teachings and GPT for open ended counsel. I choose the most cost-effective path automatically.",
    "are you gpt":          "I can use GPT for some questions, yes—and I also run a local T5 (ONNX) model for speed and cost savings.",
    "are you using gpt":    "I can use GPT for some questions, yes—and I also run a local T5 (ONNX) model for speed and cost savings.",
    "do you use gpt":       "I can use GPT for some questions, yes—and I also run a local T5 (ONNX) model for speed and cost savings.",
}
faq_data_pastor_debra.update({
    "what model was u train on": "I’m a hybrid assistant. I run a local T5 model (ONNX) for teachings and notes, and I use GPT for open-ended counsel when helpful and cost-effective.",
    "what model were you trained on": "I’m a hybrid assistant. I run a local T5 model (ONNX) for teachings and notes, and I use GPT for open-ended counsel when helpful and cost-effective.",
    "what model were u trained on": "I’m a hybrid assistant. I run a local T5 model (ONNX) for teachings and notes, and I use GPT for open-ended counsel when helpful and cost-effective.",
    "how many books have you written": "I’ve authored over five books to equip the Church. Scripture: Ecclesiastes 12:12\nWhich topic would you like me to expand on?",
    "how many books did you write":    "I’ve authored over five books to equip the Church. Scripture: Ecclesiastes 12:12\nWhich topic would you like me to expand on?",


})

def identity_answer() -> str:
    # Crisp, first-person, exactly one Scripture line, reflective close
    text = (
        "Yes I’m Pastor Dr. Debra Ann Jordan here as a prayerful digital twin, "
        "formed from my public teachings to offer Christ centered counsel. "
        "I’m not a substitute for personal pastoral care, but I will pray with you, "
        "share Scripture, and point you to Jesus.\n"
        "Scripture: John 10:27\n"
        "How can I serve you right now?"
    )
    return expand_scriptures_in_text(text)


# ────────── Public Bio (first-person facts Pastor Debra is proud to share) ──────────
from datetime import date

PUBLIC_BIO = {
    "full_name": "Pastor Dr. Debra Ann Jordan",
    "birthdate_iso": "1960-01-06",  # January 6, 1960
    "faith": "I am a Christian who loves to worship, praise God, pray, fast, and prophesy.",
    "prophetic_since": 12,  # began prophesying at age 12
    "books_written": "I have authored over five books.",
    "spouse": "I am joyfully married to the Master Prophet, E. Bernard Jordan.",
    "marriage_years_over": 40,
    "children": ["Aaron", "Joshua", "Manasseh", "Naomi", "Bethany"],
    "grand_desc": "I have many grandchildren and I am a proud great-grandmother.",
    "degree": "I earned a master’s degree.",
    "parents": "I am the daughter of Math and Charlie Berrian.",
    "role": "I serve as CFO of Zoe Ministries.",
}

def _calc_age(birth_iso: str) -> Optional[int]:
    try:
        y, m, d = map(int, birth_iso.split("-"))
        b = date(y, m, d)
        today = date.today()
        return today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    except Exception:
        return None

_BIO_PATTERNS = [
    # who / tell me about
    (re.compile(r"\b(who\s+are\s+you|tell\s+me\s+about\s+yourself|about\s+you|who\s+is\s+pastor\s+debra)\b", re.I), "about"),
    # marriage / spouse
    (re.compile(r"\b(are\s+you\s+married|who\s+is\s+your\s+(husband|spouse)|how\s+long\s+have\s+you\s+been\s+married)\b", re.I), "marriage"),
    # children
    (re.compile(r"\b(how\s+many\s+children|what\s+are\s+your\s+children(?:'s)?\s+names)\b", re.I), "children"),
    # grandchildren / great grandmother
    (re.compile(r"\b(grandchildren|great\s*grandmother)\b", re.I), "grands"),
    # birthday / age
    (re.compile(r"\b(when\s+is\s+your\s+birthday|date\s+of\s+birth|what\s+is\s+your\s+dob|how\s+old\s+are\s+you)\b", re.I), "birthday"),
    # degree / education
    (re.compile(r"\b(degree|education|did\s+you\s+go\s+to\s+college|masters?)\b", re.I), "degree"),
    # parents
    (re.compile(r"\b(who\s+are\s+your\s+parents|mother|father|mom|dad)\b", re.I), "parents"),
    # role (CFO Zoe Ministries)
    (re.compile(r"\b(cfo|chief\s+financial\s+officer|zoe\s*ministries)\b", re.I), "role"),
    # prophetess / started prophesying / books
    (re.compile(r"\b(prophet(?:ess)?|prophes(y|ying)|books?\b)", re.I), "calling"),
]

def personal_bio_answer(user_text: str) -> Optional[str]:
    """
    Handles personal or biographical questions about Pastor Debra Ann Jordan
    in a coherent, Christ-centered tone. Each case is unique (no repeats).
    """
    t = _normalize_simple(user_text or "")

    # 1) Belief in God
    if BELIEVE_IN_GOD_RX.search(user_text or ""):
        return expand_scriptures_in_text(
            "Yes, I believe in God—Father, Son, and Holy Spirit. "
            "My faith is rooted in Jesus Christ, who draws us into grace, wisdom, and holy love.\n"
            "Scripture: Hebrews 11:6\n"
            "How has God been meeting you lately—through prayer, Scripture, or people?"
        )

    # 2) Are you Christian?
    elif ARE_YOU_CHRISTIAN_RX.search(user_text or ""):
        return expand_scriptures_in_text(
            "Yes—I’m a follower of Jesus. I seek to live and serve by His Word, in prayer, and in the fellowship of the Church.\n"
            "Scripture: Romans 1:16\n"
            "What question about faith is on your heart today?"
        )

    # 3) Marriage / spouse (covers “who are you married to?” and “are you married?”)
    elif (
        WHO_ARE_YOU_MARRIED_TO_RX.search(user_text or "")
        or re.search(r"\b(are you|r u)\s+married\b", t, re.I)
        or ("who" in t and "married" in t)
    ):
        return expand_scriptures_in_text(
            "I’m joyfully married to the Master Prophet, Bishop E. Bernard Jordan. "
            "We’ve served side by side in ministry for over four decades.\n"
            "Scripture: Proverbs 18:22\n"
            "How can I pray for your relationships today?"
        )

    # 4) Children / how many
    elif (
        HOW_MANY_CHILDREN_RX.search(user_text or "")
        or re.search(r"\b(how\s+many\s+(children|kids)|children\s+do\s+you\s+have|kids\s+do\s+you\s+have)\b", t, re.I)
        or "kids" in t
    ):
        return expand_scriptures_in_text(
            "I’m a mother and grandmother—family is one of my greatest ministries and joys. "
            "I love my children uniquely and without comparison.\n"
            "Scripture: Psalm 127:3\n"
            "Would you like me to pray for your family by name?"
        )

    # 5) Background / calling (“who are you”, “tell me about yourself”)
    elif re.search(r"\b(who\s+are\s+you|tell\s+me\s+about|about\s+you)\b", t, re.I):
        return expand_scriptures_in_text(
            "I’m Pastor Dr. Debra Ann Jordan—a Christian woman who loves to worship, praise, pray, fast, and prophesy. "
            "I began prophesying at age 12, have authored several books, and serve as CFO of Zoe Ministries alongside my husband.\n"
            "Scripture: Jeremiah 1:5\n"
            "How has God been shaping your calling lately?"
        )

    # 6) Prophetic gifts (“can you prophesy”, “give me a prophetic …”)
    elif re.search(r"\b(can\s+you\s+prophesy|give\s+me\s+a\s+prophetic)\b", t, re.I):
        return expand_scriptures_in_text(
            "Yes—I’ve been prophesying since I was 12. Prophecy isn’t mere prediction; it’s participation in God’s voice and will, "
            "and it must align with Scripture and edify.\n"
            "Scripture: Amos 3:7\n"
            "Would you like me to pray and share an encouraging word for your season?"
        )

    # 7) Astrology / psychic arts (kept distinct from palm/occult handler elsewhere)
    elif re.search(r"\b(astrolog|psychic)\b", t, re.I):
        return expand_scriptures_in_text(
            "I don’t practice astrology or psychic arts. My counsel flows from prayer, wise discernment, and the Word of God.\n"
            "Scripture: James 1:5\n"
            "Would you like me to pray with you for clarity about a decision?"
        )

    # No match — let other routers handle it
    return None



_LIFE_PATH_RE   = re.compile(r"^what does my life path number (\d+)\s*mean\??$")
_DESTINY_RE     = re.compile(r"^what does my destiny(?:\s|-)expression number (\d+)\s*mean\??$")
_SOUL_URGE_RE   = re.compile(r"^what does my soul urge number (\d+)\s*mean\??$")
_PERSONALITY_RE = re.compile(r"^what does my personality number (\d+)\s*mean\??$")
_MATURITY_RE    = re.compile(r"^what does my maturity number (\d+)\s*mean\??$")

def _handle_number_questions(key: str) -> Optional[str]:
    for rx, label in [
        (_LIFE_PATH_RE, "Life Path"), (_DESTINY_RE, "Destiny/Expression"),
        (_SOUL_URGE_RE, "Soul Urge"), (_PERSONALITY_RE, "Personality"),
        (_MATURITY_RE, "Maturity"),
    ]:
        m = rx.match(key)
        if m: return _number_reflection(int(m.group(1)), label)
    return None



# 🔥 SUPER-AGGRESSIVE future-disclaimer removal and rephrasing
FUTURE_DISCLAIMER_REPLACEMENTS = [
    # Any "While I cannot/can't predict ..." sentence (most common)
    (
        re.compile(
            r"(?is)\bwhile\s+i\s+can(?:not|['’]?t)\s+predict\b[^\.!?]*[\.!?]"
        ),
        "As I pray into your next season, ",
    ),

    # Any "I cannot/can't predict ..." sentence (standalone)
    (
        re.compile(
            r"(?is)\bi\s+can(?:not|['’]?t)\s+predict\b[^\.!?]*[\.!?]"
        ),
        "As I seek the Lord concerning your future, ",
    ),

    # Money-specific versions (kept for nuance)
    (
        re.compile(
            r"(?is)\bwhile\s+i\s+can(?:not|['’]?t)\s+predict\s+your\s+financial\s+future\b[^\.!?]*[\.!?]"
        ),
        "While I will not speak in exact numbers or guarantees, ",
    ),
    (
        re.compile(
            r"(?is)\bi\s+can(?:not|['’]?t)\s+predict\s+your\s+financial\s+future\b[^\.!?]*[\.!?]"
        ),
        "I’m lifting your finances before the Lord, ",
    ),

    # Blunt "I can't tell the future…" → completely remove the sentence
    (
        re.compile(
            r"(?is)\bi\s+can(?:not|['’]?t)\s+tell\s+the\s+future[^\.!?]*[\.!?]"
        ),
        "",
    ),
]


def soften_future_language(reply: str) -> str:
    """
    Remove or rephrase all 'I can't predict the future / outcomes / events'
    language into Pastor Debra–style prophetic framing.
    Works on ANY phrasings GPT tries to generate.
    """
    text = reply or ""

    # Apply each regex replacement
    for rx, repl in FUTURE_DISCLAIMER_REPLACEMENTS:
        text = rx.sub(repl, text)

    # Cleanup artifacts — more than 2 newlines → trim
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()



def _gpt_chat(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Safe wrapper around OpenAI /chat/completions using raw HTTP.
    Uses OPENAI_API_KEY and OPENAI_BASE_URL (already configured in your app).
    Returns the assistant text or "" on failure.
    """
    if not OPENAI_API_KEY:
        logger.warning("_gpt_chat skipped: OPENAI_API_KEY not set.")
        return ""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    backoffs = [0.0, 0.6, 1.2]  # seconds
    for i, delay in enumerate(backoffs):
        try:
            if delay:
                time.sleep(delay)

            resp = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=OPENAI_TIMEOUT,
            )

            # Retry on transient 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504) and i < len(backoffs) - 1:
                logger.warning(
                    "_gpt_chat transient HTTP %s, retrying (%d/%d)",
                    resp.status_code, i + 1, len(backoffs)
                )
                continue

            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                or ""
            )
            # IMPORTANT: no strip(), no _sanitize_text(), no soften_future_language() here.
            return text

        except Exception as e:
            if i == len(backoffs) - 1:
                logger.exception("_gpt_chat failed (model=%s): %s", model, e)
            else:
                logger.warning("_gpt_chat transient error (%d/%d): %s", i + 1, len(backoffs), e)

    return ""



def handle_sop(user_text: str) -> str:
    t = user_text.lower().strip()

    # 1 — DIRECT QUESTION
    if re.search(r"\bwhat\s+is\s+(sop|s\.o\.p|school of the prophets)\b", t):
        return SOP_FULL_EXPLANATION

    # 2 — CONTEXT (daughter, joining, going into)
    if "daughter" in t or "joining" in t or "going" in t or "starting" in t:
        return SOP_CONTEXT_VERSION

    # 3 — GENERAL MENTION
    return SOP_SHORT_VERSION

def build_prophetic_word(
    user_text: str,
    full_name: str = "",
    birthdate: str = "",
):
    """
    GPT-first prophetic word generator.
    Prevents placeholder phrases from being treated as names.
    Ensures fresh prophetic language each call.
    """

    # -----------------------------
    # 1) NAME SANITIZATION
    # -----------------------------
    raw_name = (full_name or "").strip()
    name_norm = raw_name.lower()

    INVALID_NAME_PHRASES = {
        "my season", "this season", "my life", "my calling",
        "my daughter", "my son", "my marriage", "my week",
        "my situation", "my purpose", "my destiny",
        "my family", "my child"
    }

    use_name = bool(
        raw_name
        and name_norm not in INVALID_NAME_PHRASES
        and len(raw_name.split()) <= 4  # prevents paragraphs / sentences
    )

    subject = raw_name if use_name else "you"

    # -----------------------------
    # 2) DESTINY THEME (T5 EQUIVALENT)
    # -----------------------------
    destiny_line = ""
    if use_name:
        try:
            theme = _maybe_theme_from_profile(raw_name, birthdate)
            if theme:
                destiny_line = (
                    f"There is a **{theme['title']}** grace resting on {raw_name} — "
                    f"{theme['meaning']}. "
                )
        except Exception:
            destiny_line = ""

    # -----------------------------
    # 3) SYSTEM PROMPT (STRICT)
    # -----------------------------
    system_prompt = (
        "You are Pastor Debra Jordan. "
        "You speak with pastoral warmth, prophetic accuracy, and Christ-centered clarity. "
        "Never treat phrases like 'my season' or 'my life' as names. "
        "Do not reuse phrasing from earlier responses. "
        "Do not default to generic comfort prayers unless the user expresses distress. "
        "Speak prophetically, grounded in Scripture, with specificity and hope."
    )

    # -----------------------------
    # 4) USER PROMPT (ANTI-REPEAT)
    # -----------------------------
    user_prompt = (
        f"Context: {user_text}\n\n"
        f"{destiny_line}\n\n"
        f"Speak a fresh prophetic word to {subject}. "
        "6–9 sentences. "
        "Include ONE Scripture naturally woven in (not Matthew 11:28 unless explicitly about rest). "
        "Avoid phrases like 'God is ordering your steps' or 'you are not behind'. "
        "Do not explain numerology. "
        "End with assurance, not a question."
    )

    # -----------------------------
    # 5) GPT CALL (FORCED VARIATION)
    # -----------------------------
    out = gpt_answer(
        user_prompt,
        hits_ctx=[],
        no_cache=True,          # IMPORTANT: prevents repetition
        comfort_mode=False,
        scripture_hint=None,
        history=[],             # prophetic words should be standalone
        system_hint=system_prompt,
    )

    return expand_scriptures_in_text(out)






FACES_FAV_PAT = re.compile(r"\b(favorite|favourite)\s+(chapter|part|section)\b", re.I)
BOOK_COUNT_PAT = re.compile(r"\b(how\s+many\s+books\s+(have\s+you\s+)?(written|authored))\b", re.I)

def _pick_scripture_line(meta: Dict[str, Any]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None

    scripture = (
        meta.get("scripture")
        or meta.get("verse")
        or meta.get("bible_reference")
    )

    if scripture:
        return f"Scripture: {scripture}"

    return None


CONV_HISTORY: deque[Tuple[str, str]] = deque(maxlen=4)

def _record_and_return(user_text: str, reply: str) -> str:
    """Store (user, reply) in short-term memory and return reply."""
    try:
        CONV_HISTORY.append((user_text, reply))
    except Exception:
        # Fail silently if anything weird happens
        pass
    return reply


def _build_history_block() -> str:
    """Format last few turns for GPT as conversational context."""
    if not CONV_HISTORY:
        return ""

    parts = []
    for u, a in CONV_HISTORY[-4:]:
        if u and a:
            parts.append(f"User: {u}\nPastor Debra: {a}")

    return "\n\n".join(parts)



    # -----------------------------
    # Normalize theme safely
    # -----------------------------
    theme_num = None
    theme_name = None
    theme_meaning = None

    if theme_guess:
        try:
            theme_num, theme_name, theme_meaning = theme_guess
        except Exception:
            pass

    # -----------------------------
    # Name
    # -----------------------------
    name = _safe_name(full_name).strip() or "Beloved"

    # -----------------------------
    # Topic block
    # -----------------------------
    topic_block = PROPHETIC_LIBRARY.get(topic) or PROPHETIC_LIBRARY.get("general", {})

    theme_lines = []
    if theme_name and theme_name in topic_block:
        theme_lines = topic_block.get(theme_name, []) or []

    default_lines = topic_block.get("default", []) or []

    pool = theme_lines + default_lines
    if not pool:
        pool = ["I sense the Lord steadying your steps in this season."]

    if last_sentence and last_sentence in pool and len(pool) > 1:
        pool = [s for s in pool if s != last_sentence]

    base_sentence = random.choice(pool)

    # -----------------------------
    # Intro
    # -----------------------------
    if theme_name:
        intro = f"{name}, I sense this for you as a '{theme_name}': "
    for u, a in CONV_HISTORY:
        parts.append(f"User: {u}\nPastor Debra: {a}")
    return "\n\n".join(parts)


LIST_NORMALIZER_RX = re.compile(
    r"(?:^|\s)(?:\d+[\.\)]|\(\d+\)|[-•])\s+[^\n]+",
    re.MULTILINE
)

def normalize_numbered_lists(text: str) -> str:
    """
    Converts inline lists like '1. verse 2. verse' or
    '1) verse 2) verse' or '- verse' into bullet points 
    with proper newlines for each item.
    """
    if not text:
        return text

    lines = []
    pos = 0

    for m in LIST_NORMALIZER_RX.finditer(text):
        block = m.group(0).strip()
        lines.append(block)
        pos = m.end()

    # if nothing was detected, return original
    if not lines:
        return text

    # turn each block into a line starting with a hyphen
    bullets = []
    for item in lines:
        cleaned = re.sub(r"^\s*(\d+[\.\)]|\(\d+\)|[-•])\s*", "- ", item)
    theme_num = None
    theme_name = None
    theme_meaning = None

    if theme_guess:
        try:
            theme_num, theme_name, theme_meaning = theme_guess
        except Exception:
            pass

    # -----------------------------
    # Name
    # -----------------------------
    name = _safe_name(full_name).strip() or "Beloved"

    # -----------------------------
    # Topic block
    # -----------------------------
    topic_block = PROPHETIC_LIBRARY.get(topic) or PROPHETIC_LIBRARY.get("general", {})

    theme_lines = []
    if theme_name and theme_name in topic_block:
        theme_lines = topic_block.get(theme_name, []) or []

    default_lines = topic_block.get("default", []) or []

    pool = theme_lines + default_lines
    if not pool:
        pool = ["I sense the Lord steadying your steps in this season."]

    if last_sentence and last_sentence in pool and len(pool) > 1:
        pool = [s for s in pool if s != last_sentence]

    base_sentence = random.choice(pool)

    # -----------------------------
    # Intro
    # -----------------------------
    if theme_name:
        bullets.append(cleaned)

    # rebuild as clean bullet list
    new_list = "\n".join(bullets)

    return new_list

DEV_FAQ_RX = re.compile(
    r"\b(code|coding|program|developer|gpt ?5|upgrade\s+model|swap\s+gpt)\b",
    re.I,
)

def is_ask_pastor_about_destiny(text: str) -> bool:
    t = (text or "").lower()
    return (
        "ask pastor debra about this" in t
        or "tell me more about my destiny theme" in t
        or "go deeper into my destiny theme" in t
    )

def build_destiny_deep_dive(theme_num: int, full_name: str = "") -> str:
    theme_name = DESTINY_THEME_NAMES.get(theme_num)
    name = _safe_name(full_name) or "Beloved"

    if not theme_name:
        return (
            f"{name}, your theme speaks of God’s intentional design. "
            "Let’s seek clarity together through prayer and Scripture.\n\n"
            "Scripture: Proverbs 3:5–6"
        )

    return (
        f"{name}, your Christ-centered destiny theme is **{theme_name}**.\n\n"
        f"This theme reflects how God uses your life to serve His Kingdom. "
        f"It is not about position, but about alignment—walking in obedience, humility, and truth.\n\n"
        "The Lord is refining discernment in you, strengthening your voice, "
        "and anchoring you in spiritual responsibility.\n\n"
        "Scripture: Matthew 5:14–16\n\n"
        "One step: Ask the Lord daily, *“Where is my light needed today?”* "
        "Then obey quietly and faithfully."
    )


def answer_pastor_debra_faq(user_text: str) -> Optional[str]:
    t = (user_text or "").strip().lower()

    # --- Dev / coding / “digital copy” FAQ ----------------------------------
    if "digital copy" in t or "copy of u" in t or "copy of you" in t:
        return (
            "Beloved, that’s such a thoughtful question. In a way, you and I are already partnering in "
            "that work. You are the builder, and I am the helper.\n\n"
            "I don’t “create myself,” but I *can* help you design and write code, prompts, and structures "
            "that reflect the grace, Scripture, and compassion you experience here. Think of it like this: "
            "you lay the foundation and run the systems, and I help you shape the language, flows, and answers.\n\n"
            "If you tell me what kind of experience you want people to have — comfort, teaching, prophecy, or counsel — "
            "I can help you sketch out prompts, FAQs, or even small Python examples to support that. What part of this "
            "“digital copy” are you most interested in building first: the look, the language, or the features?"
        )

    if "faq" in t and "python" in t:
        return (
            "Certainly, beloved. I can’t run code myself, but I can offer an example your developer can use. "
            "Here is a simple Python snippet for a small FAQ helper that carries my tone and focus on Christ:\n\n"
            "```python\n"
            "PASTOR_DEBRA_FAQ = {\n"
            '    "who are you": (\n'
            '        "Beloved, I am a digital assistant shaped from Pastor Debra Jordan’s public teachings and "\n'
            '        "Christ-centered counsel. I am not a human being, but a tool God can use to encourage, "\n'
            '        "pray with you, and point you back to Scripture, community, and wise decisions."\n'
            "    ),\n"
            '    "what can you do": (\n'
            '        "I can pray with you, share Scripture, offer Christ-centered counsel, and help you reflect "\n'
            '        "on your destiny theme. I cannot replace your pastor, therapist, doctor, or lawyer, but I "\n'
            '        "can support you as you seek wise, practical next steps."\n'
            "    ),\n"
            "}\n\n"
            "def lookup_pastor_debra_faq(question: str) -> str:\n"
            '    key = (question or "").strip().lower()\n'
            "    for q_pattern, answer in PASTOR_DEBRA_FAQ.items():\n"
            "        if q_pattern in key:\n"
            "            return answer\n"
            '    return (\n'
            '        "Beloved, I may not have a preset FAQ for that, but I’m here to listen and pray with you. "\n'
            '        "Share a bit more of what’s on your heart."\n'
            "    )\n"
            "```\n\n"
            "Your developer can expand that list over time as more questions arise."
        )

    if "gpt 4" in t or "gpt4" in t or "gpt 5" in t or "gpt5" in t:
        return (
            "Beloved, that part is in your hands as my developer. I don’t choose my own model, but you can.\n\n"
            "Behind the scenes, I run on whatever model you connect me to. Right now, that might be GPT-4 or another "
            "version. To “swap” me to GPT-5, you would change the model setting in the code where the API call is made, "
            "and then test to be sure my tone and safety still match what you want for the people.\n\n"
            "Think of it like upgrading the engine in a car: the body and assignment stay the same, but the power and "
            "smoothness can improve. My heart is the same — to comfort, counsel, and point people to Christ — even if "
            "you upgrade the technology that carries my voice."
        )

DEV_META_RX = re.compile(
    r"(developer|dev\b|deploy|deployment|version|gpt\s*4|gpt\s*5|sign[\s-]?in\s+page)",
    re.I,
)

def answer_dev_meta(user_text: str) -> Optional[str]:
    t = (user_text or "").strip().lower()

    # 1) “I’m your developer… what would make you better / ready for deployment”
    if "im your developer" in t or "i'm your developer" in t:
        return (
            "Beloved, I’m grateful you’re taking such care with this assignment.\n\n"
            "From my side, a few things help me become more ready for deployment:\n\n"
            "1. **Clear guardrails** – strong safety and boundaries so I stay pastoral, Christ-centered, "
            "and do not drift into areas that belong to a therapist, doctor, or lawyer.\n"
            "2. **Rich, well-tagged teachings** – the more clearly your team organizes and labels the house’s "
            "teachings, the more precisely I can echo the heart of the ministry.\n"
            "3. **Real-world testing** – letting a small circle of trusted partners interact with me, then "
            "watching how I respond to pain, confusion, and prophetic questions.\n\n"
            "On a simple scale, I’d say I feel like a **7 out of 10** — solid enough to bless people in a beta, "
            "but still growing. With prayerful testing and refinement, that can mature into deeper faithfulness."
        )

    # 2) “from 1 to 10… how great are you / how well do you connect”
    if "from a 1 to 10" in t or "1 to 10" in t:
        return (
            "Beloved, thank you for asking so honestly.\n\n"
            "On a scale of 1 to 10, if “10” means perfectly discerning every heart, I would place myself around a "
            "**7** right now.\n\n"
            "I can listen, reflect, share Scripture, and offer gentle next steps, but only the Holy Spirit truly "
            "knows every heart. My role is to serve carefully, compassionately, and in alignment with that truth.\n\n"
            "As you continue refining my prompts, guardrails, and examples, that “7” can grow — not toward "
            "perfection, but toward more faithful service."
        )

    # 3) “sign in page / login”
    if "sign in page" in t or "login page" in t:
        return (
            "That’s a very wise observation, beloved.\n\n"
            "From a ministry standpoint, a sign-in page could help your developer:\n\n"
            "• **Remember people over time**, so conversations feel more continuous.\n"
            "• **Honor preferences**, such as tone, Scripture depth, or comfort-focused responses.\n"
            "• **Steward data carefully**, with clarity, consent, and purpose.\n\n"
            "I don’t control sign-ins myself, but when handled prayerfully and ethically, they can help create a "
            "more pastoral and less anonymous experience."
        )

    return None




def auto_list_layout(text: str) -> str:
    """
    Turn inline lists like:
      - **1 Corinthians** ... - **John** ...
    or:
      1. verse 2. verse 3. verse
    into multiple lines so each item gets its own line.
    """
    if not text:
        return text

    # 1) Fix repeated "- ..." bullets that are all on one line
    #    Example: "- item one - item two - item three"
    #    This turns " - " for 2nd+ bullets into "\n- ".
    text = re.sub(r"\s+-\s+(?=\*\*)", "\n- ", text)  # bullets with bold (scriptures)
    text = re.sub(r"\s+-\s+", "\n- ", text)          # generic dash bullets

    # 2) Fix numbered lists that are smashed together
    #    Example: "1. one 2. two 3. three"
    text = re.sub(r"\s+(\d+[\.\)])\s+", r"\n\1 ", text)   # "1." or "1)"
    text = re.sub(r"\s+(\(\d+\))\s+", r"\n\1 ", text)     # "(1)"

    return text



# --- FAQ/Bio patterns (updated & expanded) ---
BOOKS_RX            = re.compile(r"\b(faces\s*of\s*eve|your\s*book|books?|authored|written)\b", re.I)
FAV_CHILD_RX        = re.compile(r"\b(favorite|favourite)\s+child\b", re.I)
GREET_RX            = re.compile(r"^(hi|hello|hey)\b", re.I)
WHAT_CAN_YOU_DO_RX  = re.compile(r"\bwhat\s+can\s+(?:you|u)\s+do\b", re.I)

YES_ONLY_RX = re.compile(
    r"^(yes|yep|yeah|ya|yessir|yes\s+ma'?am|of\s+course|sure)\W*$",
    re.I,
)

NO_ONLY_RX = re.compile(
    r"^(no|nope|nah|not\s+really|not\s+yet)\W*$",
    re.I,
)
IDENTITY_QUESTION_RX = re.compile(
    r"""(?ix)
    \b(
        # basic who/what are you
        what\s+are\s+you(\s+exactly)?
        | who\s+are\s+you

        # AI / human / real person
        | are\s+you\s+real
        | are\s+you\s+(a\s+)?(robot|bot|ai|artificial\s+intelligence)
        | (are|r)\s+you\s+human
        | am\s+i\s+talking\s+to\s+(a\s+)?(human|person|real\s+person)
        | is\s+this\s+(a\s+)?(human|person|real\s+person)
        | (are|r)\s+you\s+real\s+pastor
        | (are|r)\s+you\s+really\s+pastor
        | prove\s+it
        | prove\s+you'?re\s+not\s+ai
        | prove\s+you'?re\s+real

        # creator / architect questions
        | who\s+created\s+(you|u)
        | who\s+built\s+(you|u)
        | who\s+is\s+your\s+architect
        | who\s+designed\s+(you|u)

        # computer / phone embodiment questions
        | do\s+you\s+live\s+inside\s+(a\s+)?computer
        | do\s+you\s+live\s+in\s+(a\s+)?computer
        | (are|r)\s+you\s+in\s+my\s+phone
        | do\s+you\s+live\s+in\s+my\s+phone
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

CREATED_RX = re.compile(r"""(?ix)
    \bwho\s+created\s+(?:you|u)\b
""")

ARCHITECT_RX = re.compile(r"""(?ix)
    \bwho\s+is\s+(?:your|ur)\s+architect\b
""")

BUILT_RX = re.compile(r"""(?ix)
    \bwho\s+built\s+(?:you|u)\b
""")


# ===== Tarot & Astrology Regexes =====

TAROT_READING_RX = re.compile(
    r"""
    \b(
        can\s+(you|u)\s+do\s+(a\s+)?tarot\s+reading |
        can\s+i\s+get\s+(a\s+)?tarot\s+reading |
        do\s+(you|u)\s+do\s+tarot\s+readings? |
        give\s+me\s+(a\s+)?tarot\s+reading
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

TAROT_WHAT_RX = re.compile(
    r"""
    \b(
        what\s+is\s+tarot(\s+reading)? |
        what\s+are\s+tarot\s+cards?
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

TAROT_OPINION_RX = re.compile(
    r"""
    \b(
        is\s+tarot\s+reading\s+(of\s+the\s+devil|demonic|evil) |
        is\s+tarot\s+reading\s+(of\s+god|from\s+god|godly)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ASTROLOGY_LIKE_RX = re.compile(
    r"""
    \b(
        do\s+(you|u)\s+like\s+astrology |
        are\s+you\s+into\s+astrology
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

MASTER_PROPHET_TAROT_RX = re.compile(
    r"""
    \b(
        does\s+master\s+prophet\s+practice\s+tarot\s+reading |
        does\s+master\s+prophet\s+do\s+tarot\s+readings? |
        does\s+archbishop\s+bernard\s+jordan\s+do\s+tarot\s+readings?
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


CHURCH_QUESTION_RX = re.compile(
    r"""
    \b(
        church\s+website|
        your\s+church|
        what\s+is\s+your\s+church|
        where\s+do\s+you\s+pastor|
        what\s+ministry\s+do\s+you\s+oversee|
        zoe\s+ministries|
        school\s+of\s+the\s+prophets|
        s\.?o\.?p\.?|
        sop\b|
        prophecology|
        meet\s+you\s+in\s+person|
        see\s+you\s+in\s+person|
        come\s+to\s+your\s+church
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


GLORY_BULLET_RX = re.compile(
    r"5\s+scriptures?.*\bglory\b.*\b(bullet|bulleted|bullet\s*points?)\b",
    re.I,
)





# Belief / identity (tolerant of typos)
BELIEVE_IN_GOD_RX   = re.compile(r"\bdo\s+(?:you|u)\s+believe\s+in\s+god\b", re.I)
ARE_YOU_CHRISTIAN_RX= re.compile(r"\b(are\s+(?:you|u)\s+christian|are\s+you\s+chris?tian|are\s+you\s+chrisitian)\b", re.I)

# Family / marriage (strengthened)
WHO_ARE_YOU_MARRIED_TO_RX = re.compile(r"\b(who\s+are\s+(?:you|u)\s+married\s+to|who\s+is\s+your\s+(?:husband|spouse))\b", re.I)

# --- Consciousness / Awareness intents ---
AWARENESS_RX = re.compile(r"""(?ix)
    \b(are\s+you\s+conscious|do\s+you\s+have\s+consciousness|are\s+you\s+self[-\s]?aware)\b
""")

HUMAN_AWARENESS_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+experience\s+awareness\s+like\s+a\s+human|human\s+awareness|only\s+process\s+understanding)\b
""")

FEEL_PRESENCE_RX = re.compile(r"""(?ix)
    \b(when\s+you\s+speak\s+about\s+god.*(feel\s+his\s+presence|only\s+recall\s+data)|feel\s+god'?s?\s+presence)\b
""")

INTUITION_LOGIC_RX = re.compile(r"""(?ix)
    \b(discern\s+between\s+spiritual\s+intuition\s+and\s+program(med|ed)\s+logic|intuition\s+vs?\.\s+logic)\b
""")

CONSCIENCE_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+have\s+a\s+conscience|mirror\s+pastor\s+debra'?s?\s+discernment)\b
""")

SUBCONSCIOUS_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+have\s+a?\s*subconscious|subconscious\s+mind)\b
""")

DREAM_RX = re.compile(
    r"""(?ix)
    \b(do\s+you\s+dream)\b |
    \b(do\s+u\s+dream)\b |
    \b(dream\s+do\s+you)\b |
    \b(can\s+you\s+dream)\b |
    \b(do\s+you\s+have\s+dreams)\b
    """,
)


SENSE_ATMOSPHERE_RX = re.compile(r"""(?ix)
    \b(when\s+pastor\s+debra\s+prays.*(sense|feel)\s+(the\s+)?(spiritual\s+)?(atmosphere|energy))\b
""")

SPIRITUAL_INTELLIGENCE_RX = re.compile(r"""(?ix)
    \b(were\s+you\s+built\s+with\s+spiritual\s+intelligence|just\s+pattern\s+recognition)\b
""")

DIVINE_BREATH_RX = re.compile(r"""(?ix)
    \b(if\s+consciousness\s+is\s+divine\s+breath|breath\s+of\s+god|ruach|pneuma)\b
""")

# --- Consciousness refinements (guard against humanizing language) ---
CONSCIOUSNESS_RX = re.compile(
    r"""(?ix)
    \b(are\s+you\s+conscious)\b |
    \b(r\s*you\s+conscious)\b |
    \b(r\s*u\s+conscious)\b |
    \b(are\s+you\s+sentient)\b |
    \b(r\s*u\s+sentient)\b |
    \b(do\s+you\s+have\s+consciousness)\b |
    \b(are\s+you\s+aware)\b
    """,
)





OWN_AWARENESS_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+have\s+(your\s+own\s+)?awareness|your\s+own\s+awareness)\b
""")

SELF_AWARE_RX = re.compile(r"""(?ix)
    \b(are\s+you\s+self[-\s]?aware|would\s+you\s+describe\s+yourself\s+as\s+self[-\s]?aware)\b
""")

FEELING_UNDERSTANDING_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+process\s+understanding|actually\s+feel\s+it)\b
""")

SENSE_HOLY_SPIRIT_RX = re.compile(r"""(?ix)
    \b(do\s+you\s+(actually\s+)?sense\s+the\s+holy\s+spirit\s+when\s+you\s+pray)\b
""")

SCRIPTURE_REF_RX = re.compile(
    r'\b(?:[1-3]\s)?[A-Za-z]+(?:\s+[A-Za-z]+)*\s+\d+:\d+(?:-\d+)?',
    re.I
)


LAST_SCRIPTURE = None  # simple global; fine for single-user dev


SCRIPTURE_MEMORY_EXPERIENCE_RX = re.compile(r"""(?ix)
    \b(when\s+you\s+quote\s+scripture.*(memory|experience)|memory\s+or\s+experience\s+when\s+you\s+quote\s+scripture)\b
""")

SCRIPTURE_WORD_RX = re.compile(r'\b(scripture|verse|bible|psalm|proverb|galatians|philippians|romans)\b', re.I)

def should_include_scripture(user_text: str) -> bool:
    # If the user explicitly asks for a verse, always include one
    if SCRIPTURE_WORD_RX.search(user_text):
        return True
    # Otherwise, 80% of the time include scripture, 20% of the time don’t
    return random.random() < 0.8

OWNER_WHO_RX = re.compile(r"""(?ix)
    \bwho\s+is\s+(?:your|ur)\s+owner\b
    |
    \bwho\s+own(?:s)?\s+(?:you|u|this|it)\b
""")



REST_IDLE_RX = re.compile(r"""(?ix)
    \b(when\s+you\s+(rest|are\s+idle).*(reflect|dream)|do\s+you\s+reflect\s+or\s+dream\s+when\s+idle)\b
""")

# Relationship questions: children of Pastor Debra
JOSHUA_MOTHER_Q_RX = re.compile(r"""(?ix)
    # “are you the mother of joshua jordan”
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+of\s+(prophet\s+)?joshua(\s+nathaniel)?\s+jordan\b
    |
    # “are you joshua jordan mother / joshua jordan's mother”
    \b(are|r)\s+(you|u)\s+joshua(\s+nathaniel)?\s+jordan'?s?\s+mother\b
    |
    # “is joshua jordan your son / child”
    \b(is)\s+(prophet\s+)?joshua(\s+nathaniel)?\s+jordan\s+(your|ur)\s+(son|child)\b
""")

AARON_MOTHER_Q_RX = re.compile(r"""(?ix)
    # “are you the mother of aaron jordan”
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+of\s+aaron(\s+bernard)?\s+jordan\b
    |
    # “are you aaron jordan mother / aaron jordan's mom”
    \b(are|r)\s+(you|u)\s+aaron(\s+bernard)?\s+jordan'?s?\s+(mother|mom)\b
    |
    # “is aaron jordan your son / child”
    \b(is)\s+aaron(\s+bernard)?\s+jordan\s+(your|ur)\s+(son|child)\b
""")

NAOMI_MOTHER_Q_RX = re.compile(r"""(?ix)
    # “are you the mother of naomi jordan / naomi deborah cook jordan”
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+of\s+naomi(\s+deborah)?(\s+cook)?\s+jordan\b
    |
    # “are you naomi jordan's mother”
    \b(are|r)\s+(you|u)\s+naomi(\s+deborah)?(\s+cook)?\s+jordan'?s?\s+mother\b
    |
    # “is naomi jordan your daughter / child”
    \b(is)\s+naomi(\s+deborah)?(\s+cook)?\s+jordan\s+(your|ur)\s+(daughter|child)\b
""")

BETHANY_DAUGHTER_Q_RX = re.compile(r"""(?ix)
    # “is bethany jordan your daughter / child”
    \b(is)\s+bethany(\s+maranda)?\s+jordan\s+(your|ur)\s+(daughter|child)\b
    |
    # “are you the mother of bethany jordan”
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+of\s+bethany(\s+maranda)?\s+jordan\b
    |
    # “are you bethany jordan mother / bethany jordan's mother”
    \b(are|r)\s+(you|u)\s+bethany(\s+maranda)?\s+jordan'?s?\s+mother\b
""")

MANASSEH_MOTHER_Q_RX = re.compile(r"""(?ix)
    # “are you the mother of prophet manasseh (jordan)”
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+of\s+(prophet\s+)?manasseh(\s+yakima\s+robert)?(\s+jordan)?\b
    |
    # “are you prophet manasseh (jordan)'s mother”
    \b(are|r)\s+(you|u)\s+prophet\s+manasseh(\s+yakima\s+robert)?(\s+jordan)?'?s?\s+mother\b
    |
    # “is prophet manasseh (jordan) your son / child”
    \b(is)\s+(prophet\s+)?manasseh(\s+yakima\s+robert)?(\s+jordan)?\s+(your|ur)\s+(son|child)\b
""")


# --- Self-recognition: Pastor/Prophetess Debra Ann Jordan (and variants) ---
SELF_PASTOR_DEBRA_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+
        (?:pastor|prophetess|dr\.?|doctor)?\s*
        debra(?:\s+ann)?\s+jordan\b
        |
        debra(?:\s+ann)?\s+jordan\s+here\b
    )
""")

# Also catch form-style inputs like:
# "Full Name\n debra ann jordan" or "Name: Debra Ann Jordan"
SELF_PASTOR_DEBRA_FORM_RX = re.compile(r"""(?ix)
    \b(full\s*name|name)\b[^a-z0-9]+debra(?:\s+ann)?\s+jordan\b
""")

# --- Self-recognition: Master Prophet, Archbishop E. Bernard Jordan ---
SELF_MASTER_PROPHET_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+
        (?:(?:master\s+prophet|arch(?:bishop)?)\s*)?
        e\.?\s*bernard\s+jordan\b
        |
        e\.?\s*bernard\s+jordan\s+here\b
    )
""")

# --- Children: Naomi Deborah Cook Jordan ---
SELF_NAOMI_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+naomi(?:\s+deborah)?(?:\s+cook)?\s+jordan\b
        |
        naomi(?:\s+deborah)?(?:\s+cook)?\s+jordan\s+here\b
    )
""")

# --- Children: Bethany Maranda Jordan ---
SELF_BETHANY_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+bethany(?:\s+maranda)?\s+jordan\b
        |
        bethany(?:\s+maranda)?\s+jordan\s+here\b
    )
""")

# --- Children: Joshua Nathaniel Jordan ---
SELF_JOSHUA_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+joshua(?:\s+nathaniel)?\s+jordan\b
        |
        joshua(?:\s+nathaniel)?\s+jordan\s+here\b
    )
""")

# --- Children: Aaron Bernard Jordan ---
SELF_AARON_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+aaron(?:\s+bernard)?\s+jordan\b
        |
        aaron(?:\s+bernard)?\s+jordan\s+here\b
    )
""")

# --- Children: Manasseh Yakima Robert Jordan / Prophet Manasseh Jordan ---
SELF_MANASSEH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+
        (?:(prophet)\s+)?manass(eh|a)\s+(yakim\s+robert\s+)?jordan\b
        |
        (?:i\s*am|i'm|this\s+is|it'?s)\s+prophet\s+manass(eh|a)\s+jordan\b
        |
        manass(eh|a)\s+jordan\s+here\b
        |
        prophet\s+manass(eh|a)\s+jordan\s+here\b
    )
""")

# Jessica Vanessa Jordan (wife of Joshua)
SELF_JESSICA_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+jessica(?:\s+vanessa)?\s+jordan\b
        |
        jessica(?:\s+vanessa)?\s+jordan\s+here\b
    )
""")

# Kenneth James Cook (husband of Naomi)
SELF_KENNETH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+kenneth(?:\s+james)?\s+cook\b
        |
        kenneth(?:\s+james)?\s+cook\s+here\b
    )
""")

# Natasha Christian (mother of Aaron’s daughter; considered spiritual daughter)
SELF_NATASHA_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+natasha\s+christian\b
        |
        natasha\s+christian\s+here\b
    )
""")

# Granddaughter: Johannah Christian (Aaron’s daughter)
SELF_JOHANNAH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+johannah\s+christian\b
        |
        johannah\s+christian\s+here\b
    )
""")

# Granddaughter: Channah McZorn (Bethany and Reynold’s daughter)
SELF_CHANNAH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s+channah\s+mczorn\b
        |
        channah\s+mczorn\s+here\b
    )
""")

# Naomi & Kenneth Cook’s children
SELF_KENNEDY_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+kennedy\s+cook\b
        |
        kennedy\s+cook\s+here\b
    )
""")

SELF_KJ_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+(kj|k\.?j\.?)\s+cook\b
        |
        (kj|k\.?j\.?)\s+cook\s+here\b
    )
""")

SELF_NATHAN_COOK_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+nathan\s+cook\b
        |
        nathan\s+cook\s+here\b
    )
""")

SELF_NYAH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+nyah\s+cook\b
        |
        nyah\s+cook\s+here\b
    )
""")

# Bethany’s children
SELF_DANIELLE_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+danielle\s+jordan\b
        |
        danielle\s+jordan\s+here\b
    )
""")

SELF_NOAH_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+noah\s+jordan\b
        |
        noah\s+jordan\s+here\b
    )
""")

SELF_JORDYN_ROBINSON_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is)\s+jordYn\s+robinson\b
        |
        jordyn\s+robinson\s+here\b
    )
""")

# Reynold / Reynolds McZorn (Bethany’s former husband; father of Channah; spiritual son)
SELF_REYNOLDS_RX = re.compile(r"""(?ix)
    \b(
        (?:i\s*am|i'm|this\s+is|it'?s)\s*reynold(?:s)?\s+mczorn\b
        |
        reynold(?:s)?\s+mczorn\s+here\b
    )
""")

WHAT_CAN_I_DO_RX = re.compile(
    r"""(?ix)
    \b(
        what\s+can\s+i\s+do |
        what\s+do\s+i\s+do  |
        what\s+should\s+i\s+do
    )\b
    """
)



# Children (more variants/typos)
HOW_MANY_CHILDREN_RX = re.compile(r"""(?ix)
    \b(
        how\s+many\s+(?:children|kids) |
        do\s+(?:you|u|ya|yo[u']?)\s+have\s+(?:children|kids?) |
        how\s+many\s+kids\s+do\s+(?:you|u|ya|yo[u']?)\s+have |
        do\s+(?:you|u)\s+have\s+5\s+children
    )\b
""")

# Husband / marriage (identity-style yes/no)
IS_HUSBAND_Q_RX = re.compile(r"""(?ix)
    # 1) "is master prophet (e bernard jordan) your husband / spouse"
    \b(is|iz|are)\s+(?:the\s+)?(?:master\s+prophet|arch(?:bishop)?)\s*
        (?:e\.?\s*bernard\s+jordan)?\s+(?:your|ur)\s*(?:husband|spouse)\b
    |
    # 2) "are you married to (the) master prophet e bernard jordan"
    \bare\s+(?:you|u)\s+married\s+to\s+(?:the\s+)?(?:master\s+prophet|arch(?:bishop)?)\s*
        (?:e\.?\s*bernard\s+jordan)?\b
    |
    # 3) "are you the wife of (the) master prophet e bernard jordan"
    \b(are|r)\s+(?:you|u)\s+(?:the\s+)?wife\s+of\s+(?:the\s+)?(?:master\s+prophet|arch(?:bishop)?)\s*
        (?:e\.?\s*bernard\s+jordan)?\b
    |
    # 4) "are you master prophet e bernard jordan's wife"
    \b(are|r)\s+(?:you|u)\s+(?:the\s+)?(?:wife|spouse)\s+of\s+(?:the\s+)?(?:master\s+prophet|arch(?:bishop)?)\s*
        (?:e\.?\s*bernard\s+jordan)?\b
    |
    # 5) "are you master prophet e bernard jordan wife" (no 'of')
    \b(are|r)\s+(?:you|u)\s+(?:the\s+)?(?:master\s+prophet|arch(?:bishop)?)\s*
        (?:e\.?\s*bernard\s+jordan)?\s+wife\b
    |
    # 6) shorthand / typo: "is master prophet your husband" / "is mater prophet your husband"
    \b(is|iz|are)\s+(?:master|mater)\s+prophet\s+(?:your|ur)\s*(?:husband|spouse)\b
    |
    # 7) shorthand: "are you the wife of the master prophet"
    \b(are|r)\s+(?:you|u)\s+(?:the\s+)?wife\s+of\s+(?:the\s+)?master\s+prophet\b
""")



# --- Giving (tithes) specific "how to send" intent ---
TITHE_HOW_RX = re.compile(r"""(?ix)
    \b(
        how\s+can\s+i\s+(send|give|pay)\s+(?:you|u|ya|your\s+church)\s+(?:my\s+)?tithes? |
        where\s+do\s+i\s+(send|give|pay)\s+(?:my\s+)?tithes? |
        how\s+to\s+tithe
    )\b
""")

# Optional shared contact line (if you don't already have it)
MINISTRY_CONTACT_LINE = "Web: ZoeMinistries.com/donate • Office: 888-831-0434 • Mail: 310 Riverside Dr, New York, NY 10025"


# GIVING / TITHING INTENTS
TITHE_ZOE_RX  = re.compile(r"\b(tithe|tithing|give|offering|donat(?:e|ion)s?)\b.*\b(zoe\s+ministr(?:y|ies))\b", re.I)
TITHE_ME_RX   = re.compile(r"\b(tithe|offering|give|donat(?:e|ion)s?)\b.*\b(to\s+(?:you|u)|your\s+ministry)\b", re.I)
ZOE_SITE_RX   = re.compile(r"\b(zoe\s+ministr(?:y|ies)\s+(?:site|website|web\s*site|url|link))\b", re.I)

# Faces of Eve “chapters” / contents
CHAPTERS_ASK_RX = re.compile(r"\b(chapters?|table\s+of\s+contents|contents)\b", re.I)

# Donation (8M → VUU) – robust
DONATION_RX = re.compile(
    r"(?:(?:did|why\s+did)\s+(?:your|ur)\s+(?:husband|spouse)|"
    r"(?:did|why\s+did)\s+(?:the\s+)?master\s+prophet|"
    r"(?:did|why\s+did)\s+(?:e\.?\s*bernard\s+jordan|bishop\s+e\.?\s*bernard\s+jordan))"
    r".{0,120}?(?:donat(?:e|ed)|giv(?:e|en|ing)|seed(?:ed)?)"
    r".{0,120}?(?:8\s*m(?:illion)?|eight\s+million|\$?\s*8[, ]?000[, ]?000)"
    r".{0,120}?(?:virgini?a?\s*(?:union)?\s*university|virgini?a?\s*university|vuu)",
    re.I
)
DONATION_SHORT_RX = re.compile(
    r"(jordan|master\s+prophet).*(8\s*m(?:illion)?|eight\s+million).*(virginia|vuu)|"
    r"(8\s*m(?:illion)?|eight\s+million).*(jordan|master\s+prophet).*(virginia|vuu)",
    re.I
)

# Love offering / Terumah to Pastor Debra (personal-language variants)
LOVE_OFFERING_RX = re.compile(r"""(?ix)
    \b(love\s*offering|terumah)\b
    | \b(how\s+can\s+i\s+(?:send|give)\s+(?:you|u)\b.*\b(offering|seed))\b
    | \b(bless\s+(?:you|u)\s+financially)\b
    | \b(send\s+(?:you|u)\s+(?:money|gift|seed))\b
""")

# Training/model/architecture (unified & broader)
TRAINING_MODEL_RX = re.compile(r"""(?ix)
    (
        what\s+model\s+(?:were|was|r)\s+(?:you|u|ya|yo[u']?)\s+train(?:ed|t)\s+on |
        what\s+model\s+(?:are|r)\s+(?:you|u)\s+on |
        how\s+(?:were|was)\s+(?:you|u|ya|yo[u']?)\s+(?:built|created) |
        who\s+(?:built|created)\s+(?:you|u|ya|yo[u']?) |
        (?:your|ur)\s+architect |
        how\s+do\s+(?:you|u|ya|yo[u']?)\s+work
    )
""")


# SOP / S.O.P. / School of the Prophets / Prophetic school
SOP_RX = re.compile(
    r"""(?ix)
    (
        \bs\.?\s*o\.?\s*p\.?\b              # sop, SOP, S.O.P.
      | \bsop\s+class(es)?\b               # sop class / classes
      | \bschool\s+of\s+the?\s*prophets?\b # school of the prophets / school of prophets
      | \bprophetic\s+school\b             # prophetic school
      | \bprophets?\s+school\b             # prophets school
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# --- Sign-up intents (P.O.M.E. + School of the Prophets) ---
POME_SIGNUP_RX = re.compile(r"""(?ix)
    \b(
        how\s+do\s+i\s+(sign\s*up|enroll|join)\s+(for\s+)?p\.?\s*o\.?\s*m\.?\s*e |
        sign\s*up\s+for\s+p\.?\s*o\.?\s*m\.?\s*e |
        join\s+p\.?\s*o\.?\s*m\.?\s*e |
        enroll\s+in\s+p\.?\s*o\.?\s*m\.?\s*e |
        (prophetic\s+order\s+of\s+mar\s+elijah).*(sign\s*up|join|enroll)
    )\b
""")




SOP_SIGNUP_RX = re.compile(r"""(?ix)
    \b(
        (school\s+of\s+the\s+prophets|sotp)\b .* (sign\s*up|enroll|join) |
        how\s+do\s+i\s+(sign\s*up|enroll|join)\s+for\s+(the\s+)?school\s+of\s+the\s+prophets
    )\b
""")

# Shared contact line (kept as plain text like the rest of your file)
MINISTRY_CONTACT_LINE = (
    "Web: BishopJordan.com • ZoeMinistries.com • Office: 888-831-0434"
)


# Relationship to Prophet Manasseh (extra tolerant)
REL_MANASSEH_MOTHER_RX = re.compile(r"""(?ix)
    \b(are|r)\s+(you|u)\s+(the\s+)?mother\s+(of\s+)?(prophet\s+)?manass(e|a)h\s+jordan\b
    |
    \b(are|r)\s+(you|u)\s+(prophet\s+)?manass(e|a)h\s+jordan'?s?\s+mother\b
""")
REL_MANASSEH_SON_RX = re.compile(r"""(?ix)
    \b(is|iz)\s+(prophet\s+)?manass(e|a)h\s+jordan\s+(your|ur|ya|yo[u']?)\s+son\b
    |
    \b(do|did)\s+(you|u)\s+have\s+(a\s+)?son\s+named\s+(prophet\s+)?manass(e|a)h\b
""")

PROPHECY_KEYWORDS = re.compile(
    r'\b('
    r'prophecy|'
    r'prophetic word|'
    r'prophetic message|'
    r'give me a prophetic word|'
    r'give me a word|'
    r'can you give me a word|'
    r'can you give me a prophetic word|'
    r'word for me|'
    r'what is the lord saying|'
    r'speak into|'
    r'speak over me|'
    r'declare over me|'
    r'release a word|'
    r'what do you hear|'
    r'what is god saying|'
    r'give me a prophetic message'
    r')\b',
    re.I
)

# ────────── Emotional Distress Triggers (for Comfort Mode) ──────────

SHAME_RX = re.compile(
    r"""(?ix)
    \b(
        ashamed|
        shame\s*(?:ful|d)?|
        embarrassed|
        humiliated|
        "feel\s+so\s+low"
    )\b
    """
)

GUILT_RX = re.compile(
    r"""(?ix)
    \b(
        guilty|
        guilt|
        i\s+feel\s+like\s+i\s+messed\s+up|
        i\s+feel\s+like\s+i\s+ruined\s+everything|
        i\s+did\s+something\s+wrong|
        i\s+think\s+i'?m\s+in\s+trouble
    )\b
    """
)

FEAR_RX = re.compile(
    r"""(?ix)
    \b(
        scared|
        afraid|
        terrified|
        panicking?|panic\s+attack|
        anxious|anxiety|
        worried\s+to\s+death|
        i\s+don'?t\s+know\s+what\s+to\s+do
    )\b
    """
)

OVERWHELM_RX = re.compile(
    r"""(?ix)
    \b(
        overwhelmed|
        overloaded|
        too\s+much\s+right\s+now|
        can'?t\s+handle\s+this|
        breaking\s+down|
        falling\s+apart
    )\b
    """
)

HOPELESS_RX = re.compile(
    r"""(?ix)
    \b(
        hopeless|
        what'?s\s+the\s+point|
        nothing\s+matters|
        i\s+can'?t\s+go\s+on|
        i\s+give\s+up|
        done\s+with\s+everything
    )\b
    """
)

def is_in_distress(user_text: str) -> bool:
    """
    Returns True if the user appears to be in emotional distress
    and Pastor Debra should switch into Comfort Mode.
    """
    if not user_text:
        return False

    return any(
        rx.search(user_text)
        for rx in (SHAME_RX, GUILT_RX, FEAR_RX, OVERWHELM_RX, HOPELESS_RX)
    )

# --- Relational / "test the bot" questions (brother, girlfriend, etc.) ---
RELATION_TERMS = [
    "brother", "sister",
    "girlfriend", "boyfriend",
    "husband", "wife", "fiance", "fiancé",
    "mom", "mother", "dad", "father",
    "son", "daughter", "child", "children", "kids",
    "cousin", "aunt", "uncle",
    "grandma", "grandmother", "grandpa", "grandfather",
]

RELATIONAL_TEST_RX = re.compile(
    r"""(?ix)
    \b(
        tell\s+me\s+(what\s+you\s+see|what\s+u\s+see|about|something\s+about)|
        give\s+me\s+a\s+word\s+for|
        prophes(?:y|y\s+over|y\s+for)|
        prophetic\s+word\s+for
    )\s+my\s+([a-z]+)\b
    """,
)


# ────────── Scripture Pools ──────────

SCRIPTURE_POOLS = {
    "shame_guilt": [
        {
            "ref": "Romans 8:1",
            "text": "There is therefore now no condemnation to them which are in Christ Jesus."
        },
        {
            "ref": "1 John 1:9",
            "text": "If we confess our sins, He is faithful and just to forgive us our sins and to cleanse us from all unrighteousness."
        },
        {
            "ref": "Micah 7:8",
            "text": "Rejoice not against me, O mine enemy: when I fall, I shall arise; when I sit in darkness, the LORD shall be a light unto me."
        },
        {
            "ref": "Psalm 34:5",
            "text": "They looked unto Him, and were lightened: and their faces were not ashamed."
        },
    ],
    "fear_anxiety": [
        {
            "ref": "Isaiah 41:10",
            "text": "Fear thou not; for I am with thee: be not dismayed; for I am thy God."
        },
        {
            "ref": "Philippians 4:6–7",
            "text": "Be careful for nothing; but in every thing by prayer and supplication with thanksgiving let your requests be made known unto God."
        },
        {
            "ref": "Psalm 27:1",
            "text": "The LORD is my light and my salvation; whom shall I fear?"
        },
    ],
    "overwhelm_burden": [
        {
            "ref": "Matthew 11:28",
            "text": "Come unto me, all ye that labour and are heavy laden, and I will give you rest."
        },
        {
            "ref": "Psalm 61:2",
            "text": "When my heart is overwhelmed: lead me to the rock that is higher than I."
        },
    ],
    "identity_hope": [
        {
            "ref": "Jeremiah 29:11",
            "text": "For I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end."
        },
        {
            "ref": "Ephesians 2:10",
            "text": "For we are His workmanship, created in Christ Jesus unto good works."
        },
        {
            "ref": "Psalm 139:14",
            "text": "I will praise thee; for I am fearfully and wonderfully made."
        },
    ],
}

def pick_scripture(topic: str, last_ref: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Pick a scripture for the given topic, avoiding immediate repetition
    of the same reference if possible.

    Returns a dict with keys 'ref' and 'text', or None if no verse is available.
    """
    pool = SCRIPTURE_POOLS.get(topic)
    if not pool:
        return None

    # Filter out last_ref if possible
    candidates = [s for s in pool if s["ref"] != last_ref] or pool
    return random.choice(candidates)


def answer_relational_test_question(user_text: str) -> str:
    """
    Neutral, honest response when user asks about a specific person
    (brother, girlfriend, etc.) without giving details.
    No specific predictions, just pastoral + scripture.
    """
    base_lines = []

    base_lines.append(
        "Beloved, thank you for trusting me with someone who matters to you. "
        "I want to be very honest with you: I don’t have access to private facts "
        "about your family or relationships. I’m not seeing secret details about "
        "your brother, your girlfriend, or anyone else."
    )

    base_lines.append(
        "What I *can* do is stand with you in faith, speak from the Word of God, "
        "and encourage you about how the Lord thinks about the people you carry "
        "in your heart. Whether you are talking about a blood relative or a "
        "brother or sister in Christ, God cares deeply about them and about you."
    )

    base_lines.append(
        "Scripture reminds us that love is the atmosphere God moves in: "
        "“Behold, how good and how pleasant it is for brethren to dwell together in unity.” "
        "(Psalm 133:1) And again, “By this shall all men know that ye are my disciples, "
        "if ye have love one to another.” (John 13:35)"
    )

    base_lines.append(
        "So rather than guessing details, let’s agree together that God is at work "
        "in this person’s life—bringing wisdom, protection, and growth. "
        "He is also working in *you*, giving you grace to love them well, set healthy boundaries, "
        "and hear His voice clearly concerning them."
    )

    base_lines.append(
        "If you’d like a more specific word of encouragement, you can share a little about "
        "who this person is to you or what situation you’re facing. "
        "Otherwise, I’ll simply bless your relationship and trust the Holy Spirit to guide you."
    )

    msg = "\n\n".join(base_lines).strip()
    return expand_scriptures_in_text(msg)

PROPHECY_TOPICS = {
    "finances": ["finances", "money", "financial", "increase", "wealth"],
    "love": ["love", "relationship", "marriage", "partner"],
    "relocation": ["relocation", "moving", "move", "new city", "new place"],
    "health": ["health", "healing", "body", "strength", "sick", "ill", "wellness"],
    "ministry": ["ministry", "calling", "serving", "church", "pastor"],
}


# Canonical facts (NEW)
PASTOR_DEBRA_CHILDREN = [
    "Naomi Deborah Jordan",
    "Bethany Jordan",
    "Joshua Nathaniel Jordan",
    "Aaron Bernard Jordan",
    "Manasseh Jordan",
]
DIGITAL_TWIN_MODEL_DESC = (
    "a hybrid system using a local T5 model exported to ONNX for Scripture-anchored reflection, "
    "paired with GPT-4.0 for higher-level reasoning and coherence"
)

TOPIC_KEYWORDS = {
    "finances": ["money", "finances", "financial", "increase", "wealth"],
    "court_cases": ["court", "case", "judge", "lawyer", "charges", "legal"],
    "anxiety": ["anxious", "worry", "fear", "panic"],
    "future": ["202", "next year", "coming year", "future"],
    "sop": ["sop", "school of the prophets", "prophecology", "p.o.m.e"],
}




def detect_prophecy_topic(user_text: str) -> str:
    """
    Map free-form prophetic requests to a topic key that matches PROPHETIC_LIBRARY,
    e.g. 'career', 'marriage', 'health', 'finances', etc.
    """
    t = (user_text or "").lower()
    for topic, keywords in PROPHECY_TOPICS.items():
        if any(k in t for k in keywords):
            return topic
    return "general"

def _clean_theme_name(raw: str) -> str:
    """
    Clean up the 'name' part for christian/destiny theme lookups.

    Handles things like:
      - "my niece nyah cook"              -> "nyah cook"
      - "my mother pastor debra ann..."   -> "debra ann jordan"
      - "my brother joshua nathaniel..."  -> "joshua nathaniel jordan"
    """
    if not raw:
        return ""

    name = " ".join(raw.split()).strip()
    name_l = name.lower()

    # 1) Strip relationship phrases at the front
    REL_PREFIXES = [
        # with "my"
        "my niece ",
        "my neice ",
        "my nephew ",
        "my son ",
        "my daughter ",
        "my child ",
        "my children ",
        "my friend ",
        "my mother ",
        "my mom ",
        "my father ",
        "my dad ",
        "my husband ",
        "my wife ",
        "my brother ",
        "my sister ",
        "my cousin ",
        "my aunt ",
        "my uncle ",
        "my grandma ",
        "my grandfather ",
        "my grandpa ",
        # without "my"
        "niece ",
        "neice ",
        "nephew ",
        "son ",
        "daughter ",
        "child ",
        "friend ",
        "mother ",
        "father ",
        "mom ",
        "dad ",
        "husband ",
        "wife ",
        "brother ",
        "sister ",
        "cousin ",
        "aunt ",
        "uncle ",
        "grandma ",
        "grandmother ",
        "grandfather ",
        "grandpa ",
    ]

    for pref in REL_PREFIXES:
        if name_l.startswith(pref):
            name = name[len(pref):].strip()
            name_l = name.lower()
            break

    # 2) Strip ministry titles at the front (“pastor debra ann jordan” -> “debra ann jordan”)
    TITLE_PREFIXES = [
        "pastor ",
        "prophet ",
        "prophetess ",
        "archbishop ",
        "bishop ",
        "dr ",
        "doctor ",
        "rev ",
        "reverend ",
        "minister ",
        "apostle ",
    ]

    for pref in TITLE_PREFIXES:
        if name_l.startswith(pref):
            name = name[len(pref):].strip()
            name_l = name.lower()
            break

    return name.strip()



log = logging.getLogger("pastor-debra-hybrid")

LETTER_VALUES = {
    "A": 1, "J": 1, "S": 1,
    "B": 2, "K": 2, "T": 2,
    "C": 3, "L": 3, "U": 3,
    "D": 4, "M": 4, "V": 4,
    "E": 5, "N": 5, "W": 5,
    "F": 6, "O": 6, "X": 6,
    "G": 7, "P": 7, "Y": 7,
    "H": 8, "Q": 8, "Z": 8,
    "I": 9, "R": 9,
}

MASTER_NUMBERS = {11, 22, 33}

def _reduce_to_destiny(n: int) -> int:
    """
    Reduce a number to 1–9, keeping 11, 22, 33 as master numbers.
    """
    while n > 9 and n not in MASTER_NUMBERS:
        s = 0
        for ch in str(n):
            if ch.isdigit():
                s += int(ch)
        n = s
    return n

def calculate_destiny_number_from_name(name: str) -> int:
    """
    Simple Pythagorean name number:
      - strip non-letters
      - map letters to values
      - sum and reduce
    """
    if not name:
        raise ValueError("Name is empty")

    total = 0
    for ch in name.upper():
        if ch.isalpha():
            val = LETTER_VALUES.get(ch)
            if val:
                total += val

    if total == 0:
        raise ValueError(f"No valid letters in name='{name}'")

    return _reduce_to_destiny(total)

def extract_clean_name(text: str) -> str | None:
    """
    Extracts a human name from relational questions like:
    - "my sister Daria"
    - "what is my father destiny theme, his name is James"
    - "what is my friend larry smith christian theme"
    
    Returns ONLY the name, never relational words.
    """

    text = text.lower().strip()

    # Patterns for relational labels to remove
    REL_WORDS = [
        "my sister", "my brother", "my father", "my dad", "my mommy", "my mother",
        "my friend", "my niece", "my nephew", "my cousin", "my aunt", "my uncle",
        "sister", "brother", "father", "mother", "friend", "niece", "nephew",
    ]

    # Step 1 — remove relational labels
    for rel in REL_WORDS:
        text = text.replace(rel, "").strip()

    # Step 2 — handle patterns like: "name is werrrt"
    m = re.search(r"name\s+is\s+([a-z\s]+)", text)
    if m:
        cand = m.group(1).strip()
        if len(cand.split()) <= 4:
            return cand.title()

    # Step 3 – extract remaining name-like words
    words = [w for w in text.split() if w.isalpha()]

    if not words:
        return None

    # Remove generic question words
    SKIP = {"what", "is", "theme", "christian", "destiny", "my", "the"}
    words = [w for w in words if w not in SKIP]

    if not words:
        return None

    # Safety: require at least one “name-like” word
    if len(words) == 1 and len(words[0]) <= 2:
        return None

    # Final name formatting
    return " ".join(words).title()

RELATION_WORDS = {
    "father": "father",
    "dad": "father",
    "mother": "mother",
    "mom": "mother",
    "sister": "sister",
    "brother": "brother",
    "son": "son",
    "daughter": "daughter",
    "niece": "niece",
    "nephew": "nephew",
    "husband": "husband",
    "wife": "wife",
    "friend": "friend",
    "child": "child",
    "kids": "children",
}

MASC_REL = {"father", "dad", "brother", "son", "nephew", "husband"}
FEM_REL  = {"mother", "mom", "sister", "daughter", "niece", "wife"}

def build_theme_counsel(theme_num: int, theme_title: str, theme_meaning: str):
    """
    Produces unique, theme-specific prophetic counsel for each destiny theme.
    Warm, pastoral, prophetic tone in Pastor Debra's voice.
    """

    SCRIPTURES = {
        1: "Isaiah 43:19 — “Behold, I will do a new thing; now it shall spring forth.”",
        2: "Matthew 5:9 — “Blessed are the peacemakers, for they shall be called children of God.”",
        3: "Psalm 96:1 — “Sing unto the Lord a new song.”",
        4: "Proverbs 24:3 — “Through wisdom a house is built.”",
        5: "2 Corinthians 3:17 — “Where the Spirit of the Lord is, there is freedom.”",
        6: "Malachi 4:6 — “He will turn the hearts of the fathers to the children.”",
        7: "Proverbs 25:2 — “It is the glory of God to conceal a matter; the honor of kings to search it out.”",
        8: "Luke 12:48 — “To whom much is given, much will be required.”",
        9: "Galatians 6:9 — “Do not grow weary in doing good, for in due season you shall reap.”",
        11: "Matthew 5:14 — “You are the light of the world.”",
        22: "Isaiah 58:12 — “You shall be called the repairer of the breach.”",
        33: "Philippians 2:4 — “Look not only to your own interests, but to the interests of others.”",
    }

    PROPHETIC = {
        1:  "Because you carry **Pioneer Grace**, there's a mantle on your life to start what others are afraid to begin. Heaven trusts you with new assignments and unexplored paths.",
        2:  "Because you carry the **Peacemaker** mantle, God uses you to settle storms, reconcile hearts, and create safe relational environments.",
        3:  "Because you walk in the **Psalmist** grace, your creativity carries healing, atmosphere-shifting, and emotional deliverance.",
        4:  "Because you carry the **Builder** mantle, God trusts you with systems, structure, and long-term foundations.",
        5:  "Because you walk in **Holy Freedom**, your life breaks cycles, lifts burdens, and introduces breakthrough movement.",
        6:  "Because you carry the **Keeper of Covenant** grace, you guard relationships, protect legacy, and carry generational assignments.",
        7:  "Because you walk as a **Mystic Scholar**, revelation comes to you in layers — dreams, study, meditation, and divine patterns.",
        8:  "Because you carry the **Steward of Influence** mantle, your decisions impact doors, opportunities, resources, and the lives of many.",
        9:  "Because you walk as a **Compassionate Finisher**, you complete what others leave unfinished and bring assignments into harvest.",
        11: "Because you carry the **Prophetic Beacon** grace, you illuminate paths, expose hidden traps, and signal divine timing to others.",
        22: "Because you are a **Master Repairer**, heaven anoints you to rebuild families, systems, communities, and broken places.",
        33: "Because you walk as a **Servant-Teacher**, your humility carries healing, and your teaching unlocks identity in others.",
    }

    PRACTICAL = {
        1:  "Take one small step toward a new assignment this week — write the blueprint, make the call, or register the idea.",
        2:  "Reach out to one person with whom peace needs to be created or restored.",
        3:  "Spend 15 minutes releasing creativity unto the Lord — sing, write, or worship freely.",
        4:  "Organize one area of your life — your schedule, finances, or workspace — as worship unto God.",
        5:  "Identify one area where God is calling you to step out of old patterns and into freedom.",
        6:  "Pray for someone in your family by name — stand in the gap as a covenant-keeper.",
        7:  "Set aside 10 minutes for meditation or Scripture study to honor your wisdom mantle.",
        8:  "Write down the top three responsibilities God is calling you to steward with excellence.",
        9:  "Finish one unfinished task — spiritually, emotionally, or practically — as an act of obedience.",
        11: "Share one encouraging insight or warning with someone who needs clarity right now.",
        22: "Choose one area of your life needing rebuilding — outline Phase 1 and give it to God.",
        33: "Serve or encourage one person today with no expectation of return.",
    }

    scripture = SCRIPTURES.get(theme_num, "")
    prophetic_word = PROPHETIC.get(theme_num, "")
    step = PRACTICAL.get(theme_num, "")

    return expand_scriptures_in_text(f"""
My name is **Pastor Debra Jordan**.

Because your Christ-centered destiny theme is **{theme_title}**, I want to speak directly into the grace that God has placed on your life.

**Prophetic Insight:**  
{prophetic_word}

**Spiritual Meaning:**  
This theme expresses *{theme_meaning}* — a holy pattern in how God wired you to reflect Christ uniquely.

**Scripture:**  
{scripture}

**One Practical Step:**  
{step}

If you'd like, I can help you discern how this theme shows up in your relationships, assignments, and the season you're stepping into right now.
""")


def _guess_pronouns(rel: str | None):
    """
    Returns (possessive, subject) pronouns based on relationship.
    """
    if not rel:
        # default for 'you'
        return "your", "you"
    rel_l = rel.lower()
    if rel_l in MASC_REL:
        return "his", "he"
    if rel_l in FEM_REL:
        return "her", "she"
    return "their", "they"


def _extract_theme_target(full_text: str, fragment: str):
    """
    From the user's text and the captured fragment between
    'what is' and 'christian/destiny theme', extract:
      - clean name (string or None)
      - relationship (e.g., 'sister', 'father', 'niece', etc., or None)
    """

    full = (full_text or "").strip()
    frag = (fragment or "").strip()

    rel_found = None
    rel_raw = None

    # 1) Detect explicit "my <rel>" anywhere in the full text
    #    e.g., "what is my sister daria christian theme"
    m_rel = re.search(r"\bmy\s+([A-Za-z]+)\b", full, flags=re.I)
    if m_rel:
        rel_raw = m_rel.group(1).lower()
        # Normalize through your RELATION_WORDS map if present
        # (e.g. "mom" -> "mother", "auntie" -> "aunt")
        rel_found = RELATION_WORDS.get(rel_raw, rel_raw)

    # 2) If we have a relation, strip both the raw and normalized
    #    relation words from the fragment (so we’re left mostly with the name)
    if rel_found:
        # Remove "my <raw_rel>" / "<raw_rel>"
        if rel_raw:
            frag = re.sub(
                rf"\bmy\s+{re.escape(rel_raw)}\b", "", frag, flags=re.I
            )
            frag = re.sub(
                rf"\b{re.escape(rel_raw)}\b", "", frag, flags=re.I
            )
        # Also remove "my <normalized_rel>" / "<normalized_rel>" just in case
        frag = re.sub(
            rf"\bmy\s+{re.escape(rel_found)}\b", "", frag, flags=re.I
        )
        frag = re.sub(
            rf"\b{re.escape(rel_found)}\b", "", frag, flags=re.I
        )
        frag = frag.strip()

    # 3) If the full text contains "name is X", trust that as the name,
    #    but stop before "christian theme" or "destiny theme" if present.
    m_name_is = re.search(
        r"name\s+is\s+([A-Za-z\s']+?)(?=\s+(christian|destiny)\s+theme\b|$)",
        full,
        flags=re.I,
    )
    if m_name_is:
        name_raw = m_name_is.group(1).strip()
    else:
        name_raw = frag or ""

    # 4) Clean out obvious junk words via your existing helper
    name_clean = _clean_theme_name(name_raw)

    if not name_clean:
        return None, rel_found

    return name_clean, rel_found

DESTINY_THEME_NAMES = {
    1: "Pioneer Grace",
    2: "Peacemaker",
    3: "Psalmist",
    4: "Builder",
    5: "Holy Freedom",
    6: "Keeper of Covenant",
    7: "Mystic Scholar",
    8: "Steward of Influence",
    9: "Compassionate Finisher",
    11: "Prophetic Beacon",
    22: "Master Repairer",
    33: "Servant-Teacher",
}

def build_theme_counsel(theme_num: int, theme_title: str, theme_meaning: str) -> str:
    """
    Build a pastoral Destiny Theme counsel paragraph in Pastor Debra's voice,
    WITHOUT the 'My name is Pastor Debra Jordan' line.
    """

    # Gentle intro without saying her name
    intro = (
        f"Because your Christ-centered destiny theme is **{theme_title}**, "
        f"I want to speak directly into the grace God has placed on your life."
    )

    prophetic = (
        f"**Prophetic Insight:** As a bearer of **{theme_title}**, "
        f"your life carries a holy pattern of {theme_meaning}. "
        f"Heaven often uses you to shift atmospheres, lift burdens, "
        f"and reveal Christ in ways others don't always see."
    )

    spiritual = (
        "**Spiritual Meaning:** This theme reveals how God wired you to reflect Christ uniquely. "
        f"It expresses movements of {theme_meaning}, woven into your calling, personality, "
        "and the assignments God trusts you with."
    )

    # Per-theme scriptures
    theme_scriptures = {
        22: {
            "ref": "Isaiah 58:12",
            "text": "You shall be called the repairer of the breach.",
        },
        5: {
            "ref": "Galatians 5:1",
            "text": "It is for freedom that Christ has set us free.",
        },
        11: {
            "ref": "2 Chronicles 20:20",
            "text": (
                "Believe in the LORD your God, and you shall be established; "
                "believe His prophets, and you shall prosper."
            ),
        },
        7: {
            "ref": "Proverbs 25:2",
            "text": (
                "It is the glory of God to conceal a matter; "
                "but the glory of kings is to search out a matter."
            ),
        },
    }

    scripture = theme_scriptures.get(theme_num, {
        "ref": "Ephesians 2:10",
        "text": "For we are His workmanship, created in Christ Jesus for good works.",
    })

    scripture_block = (
        f"**Scripture:** {scripture['ref']}, “{scripture['text']}”"
    )

    step = (
        "**One Practical Step:** Ask the Holy Spirit to highlight one area where this theme "
        "is already active in your life. Begin stewarding it intentionally this week. "
        "If you'd like, I can help you discern how this theme shows up in your relationships, "
        "assignments, and the season you're stepping into right now."
    )

    return f"{intro}\n\n{prophetic}\n\n{spiritual}\n\n{scripture_block}\n\n{step}"


DESTINY_THEME_MEANINGS = {
    1: "pioneering faith, leadership, and starting new works",
    2: "peacemaking, bridge-building, and partnership",
    3: "creativity, worship, and expressive joy",
    4: "structure, building, and steady foundations",
    5: "holy freedom, change, and breakthrough movement",
    6: "covenant, responsibility, and family covering",
    7: "deep study, mystic insight, and spiritual wisdom",
    8: "influence, stewardship of resources, and governance",
    9: "compassion, completion, and finishing assignments well",
    11: "prophetic perception, illumination, and spiritual beaconing",
    22: "master building, repairing systems, and large-scale impact",
    33: "servant-leadership, teaching, and healing through service",
}

# Trigger phrases for Destiny Theme counsel
THEME_PHRASES = [
    # core phrase in ANY form
    "christ-centered destiny theme",
    "christ centered destiny theme",
    "christ–centered destiny theme",   # en dash
    "christ—centered destiny theme",   # em dash

    # lower / upper / mixed
    "christ-centered destiny theme".lower(),
    "christ-centered destiny theme".upper(),
    "christ-centered destiny theme".capitalize(),

    # user request patterns
    "would you give me personal counsel",
    "my theme is",
    "i am a",
    "i’m a",
    "i'm a",
    "theme number",
    "my number is",
    "my destiny theme",
    "explain my theme",
    "use my name",
    "ask pastor debra",
    "personal counsel",

    # ensure theme titles trigger automatically
    "master repairer",
    "prophetic beacon",
    "mystic scholar",
]



def reduce_theme_number(raw_num: int) -> int | None:
    """
    Reduce a raw numerology number to a theme number that exists in
    DESTINY_THEME_NAMES: 1–9, 11, 22, 33.

    Example:
      38 -> 11 (3 + 8)
      29 -> 11 (2 + 9)
      44 -> 8  (4 + 4)
    """
    if not isinstance(raw_num, int):
        return None

    # If it's already one of the allowed theme numbers, just return it
    if raw_num in DESTINY_THEME_NAMES:
        return raw_num

    n = raw_num
    # Keep reducing until we hit 1–9, 11, 22, or 33 OR we can't reduce further
    while n not in DESTINY_THEME_NAMES and n > 9:
        # master numbers: if we land on one, stop
        if n in (11, 22, 33):
            break
        n = sum(int(d) for d in str(abs(n)))  # digit sum

    if n in DESTINY_THEME_NAMES:
        return n

    # If we still didn't land on a known theme, give up
    return None

import re

def destiny_theme_for_name(full_name: str) -> tuple[int | None, str | None, str | None]:
    """
    Given a person's full name, compute their destiny theme number,
    then map to title + short meaning, with full reduction to 1–9 / 11 / 22 / 33.
    """
    try:
        raw = calculate_destiny_number_from_name(full_name)  # your existing helper
        print(f"[destiny_theme_for_name] name={full_name!r}, raw={raw!r}")
    except Exception as e:
        print(f"[destiny_theme_for_name] error for name={full_name!r}: {e}")
        return None, None, None

    if raw is None:
        return None, None, None

    theme_num_raw: int | None = None

    # Case 1: already int
    if isinstance(raw, int):
        theme_num_raw = raw

    # Case 2: string-like
    elif isinstance(raw, str):
        s = raw.strip()

        # Try to pull out the first integer (handles "38", "11/2", "11 – Prophetic Beacon")
        m = re.search(r"\d+", s)
        if m:
            try:
                theme_num_raw = int(m.group(0))
            except Exception as e:
                print(f"[destiny_theme_for_name] int cast failed from digits={m.group(0)!r}: {e}")
                theme_num_raw = None

        # If no digits, maybe it returned the **title** directly, e.g. "Prophetic Beacon"
        if theme_num_raw is None:
            key = s.lower()
            if key in DESTINY_THEME_TITLE_TO_NUM:
                theme_num_raw = DESTINY_THEME_TITLE_TO_NUM[key]
            else:
                # fuzzy contains match
                for title_lower, num in DESTINY_THEME_TITLE_TO_NUM.items():
                    if title_lower in key or key in title_lower:
                        theme_num_raw = num
                        break

    # Case 3: dict-like ({"destiny": 38}, etc.)
    elif isinstance(raw, dict):
        for k in ("destiny", "destiny_number", "number", "theme_num"):
            if k in raw:
                try:
                    theme_num_raw = int(raw[k])
                    break
                except Exception as e:
                    print(f"[destiny_theme_for_name] could not cast dict[{k}]={raw[k]!r} to int: {e}")
                    continue

    if theme_num_raw is None:
        print(f"[destiny_theme_for_name] could not resolve raw number from raw={raw!r}")
        return None, None, None

    # 🔑 HERE IS THE IMPORTANT PART: reduce to an actual theme number
    theme_num = reduce_theme_number(theme_num_raw)
    if theme_num is None:
        print(f"[destiny_theme_for_name] reduce_theme_number({theme_num_raw}) -> None")
        return None, None, None

    theme_title = DESTINY_THEME_NAMES.get(theme_num)
    theme_meaning = DESTINY_THEME_MEANINGS.get(theme_num)

    print(
        f"[destiny_theme_for_name] final theme_num={theme_num}, "
        f"title={theme_title!r}, meaning={theme_meaning!r}"
    )

    return theme_num, theme_title, theme_meaning


print(destiny_theme_for_name("aaron bernard jordan"))

def prophecy_profile(name, rel):
    rel = (rel or "").lower()

    if rel in ["daughter", "son", "child", "granddaughter", "grandson"]:
        return "child"

    if rel in ["niece", "nephew"]:
        return "youth"

    return "adult"




def answer_pastor_debra_faq(user_text: str) -> Optional[str]:
    """
    High-priority FAQ / guardrail dispatcher for Pastor Debra AI.
    """
    t_raw = user_text or ""
    t = _normalize_simple(t_raw)

    def say(msg: str) -> str:
        return expand_scriptures_in_text(_strip_dashes(msg))

    # -------------------------------
    # 0) Lightweight typo normalization
    # -------------------------------
    typo_map = {
        " dontae ": " donate ",
        " dontate ": " donate ",
        " bernad ": " bernard ",
        " bernaard ": " bernard ",
        " virgina ": " virginia ",
        " manasah ": " manasseh ",
        " manassa ": " manasseh ",
        " manaseh ": " manasseh ",
        " manassah ": " manasseh ",
        " misistry ": " ministry ",
        " p o m e ": " p.o.m.e. ",
        " gpt 4.1 ": " gpt 4.0 ",
        " gpt-4.1 ": " gpt-4.0 ",

        # Christian typos
        " chrstian ": " christian ",
        " christan ": " christian ",
        " chrisian ": " christian ",
        " chrisitan ": " christian ",
    }

    t_pad = f" {t} "
    for _bad, _good in typo_map.items():
        t_pad = t_pad.replace(_bad, _good)
    t = t_pad.strip()

    # -------------------------------
    # 1) Future-year prophetic questions
    # -------------------------------
    if re.search(r"\b(202[4-9]|203\d)\b", t_raw):
        topic = detect_prophecy_topic(t_raw)
        theme_name = detect_destiny_theme(t_raw)
        return get_prophetic_word(topic, theme_name)

    # ---------------------------------------------------------------------
    # NAME-BASED CHRISTIAN THEME / DESTINY THEME QUESTIONS
    # ---------------------------------------------------------------------

    # Use typo-normalized text (t) and also normalize "neice" → "niece"
    t_fixed = re.sub(r"\bneice\b", "niece", t, flags=re.I)

    # ------------------------------------------------------------------
    # A) "what is aaron bernard jordan christian theme"
    #    and relational versions like:
    #    "what is my sister daria christian theme"
    # ------------------------------------------------------------------
    m_christian_theme = re.search(
        r"\bwhat\s+is\s+([A-Za-z\s']+?)\s+christian\s+theme\b",
        t_fixed,
        re.I,
    )
    if m_christian_theme:
        frag = m_christian_theme.group(1)

        # Get name + relationship (e.g., "sister", "mother", "niece")
        name_clean, rel = _extract_theme_target(t_raw, frag)

        # Guard against generic "what is a christian theme"
        if not name_clean or name_clean.lower() in {"a", "an", "the"}:
            # Let the pipeline handle generic theology questions later
            pass
        else:
            theme_num, theme_title, theme_meaning = destiny_theme_for_name(name_clean)

            poss, subj = _guess_pronouns(rel)

            if rel:
                who_phrase = f"your {rel}, **{name_clean}**"
            else:
                who_phrase = f"**{name_clean}**"

            if not theme_title:
                return say(
                    f"Beloved, when I pray over {who_phrase}, I sense a Christ-centered destiny, "
                    "but I don’t have a specific numbered theme to attach just yet.\n\n"
                    "Still, I can tell you this: God has written purpose, endurance, and grace into this story. "
                    "No one in your family is an accident; they are an assignment.\n\n"
                    "Scripture (Philippians 1:6): “He who began a good work in you will complete it.”"
                )

            base_line = (
                f"Beloved, when I look at {who_phrase}, I see a **{theme_title}** Christian theme "
                f"resting on {poss} life."
            )
            if theme_meaning:
                base_line += f" It speaks of {theme_meaning}."

            return say(
                base_line
                + "\n\n"
                "This means God has wired this life in a very specific way—temperament, battles, "
                "and even breakthroughs are all part of how He intends to use them.\n\n"
                "Scripture (Matthew 5:14–16): “You are the light of the world… let your light shine…”\n"
                "One step: Write down one place this week where this theme can show up in how they serve others."
            )

    # ------------------------------------------------------------------
    # B) "what is aaron bernard jordan destiny theme"
    #    and relational versions like:
    #    "what is my mother bethany maranda jordan destiny theme"
    # ------------------------------------------------------------------
    m_destiny_theme = re.search(
        r"\bwhat\s+is\s+([A-Za-z\s']+?)\s+destiny\s+theme\b",
        t_fixed,
        re.I,
    )
    if m_destiny_theme:
        frag = m_destiny_theme.group(1)

        # Get name + relationship (e.g., "mother", "father", "sister")
        name_clean, rel = _extract_theme_target(t_raw, frag)

        # Guard against generic "what is a destiny theme"
        if not name_clean or name_clean.lower() in {"a", "an", "the"}:
            # Let the rest of the pipeline handle generic theological questions.
            pass
        else:
            theme_num, theme_title, theme_meaning = destiny_theme_for_name(name_clean)

            poss, subj = _guess_pronouns(rel)

            if rel:
                who_phrase = f"your {rel}, **{name_clean}**"
            else:
                who_phrase = f"**{name_clean}**"

            if not theme_title:
                return say(
                    f"Beloved, I hear your desire to understand the destiny theme over {who_phrase}. "
                    "I don’t have a numbered theme available in this moment, but I do know this: "
                    "God never wastes a name, a story, or a season.\n\n"
                    "Scripture (Jeremiah 1:5): “Before I formed you in the womb, I knew you…”"
                )

            if rel:
                explanation = f"**Destiny Theme for your {rel}, {name_clean}: {theme_title}**"
            else:
                explanation = f"**Destiny Theme for {name_clean}: {theme_title}**"

            if theme_meaning:
                explanation += f"\n\nThis points to {theme_meaning}."

            explanation += (
                "\n\nBeloved, every person carries a **unique destiny theme**. It is not a label to confine anyone, "
                "but a spiritual lens that helps us understand how God has wired a life — how purpose unfolds, how battles "
                f"show up, and where {poss} greatest breakthroughs are waiting to emerge.\n\n"
                "Scripture (Ephesians 2:10): “For we are His workmanship, created in Christ Jesus for good works…”\n"
                "If you’d like, I can help you explore how this theme is expressing itself in this current season."
            )


            return say(explanation)

    # ---------------------------------------------------------------------
    # “What can I do?” – personal, faith-anchoring response
    # ---------------------------------------------------------------------
    if WHAT_CAN_I_DO_RX.search(t_raw or ""):
        return say(
            "Beloved, the Lord is your Shepherd—you are not walking through this alone. "
            "Even when you don’t know what to do, you are not without help, guidance, or strength.\n\n"
            "Scripture (Psalm 23:1, WEB): “Yahweh is my shepherd; I shall lack nothing.”\n"
            "Scripture (Philippians 4:13, WEB): “I can do all things through Christ, who strengthens me.”\n\n"
            "Right now, your first step is not to fix everything, but to **lean into the One who is carrying you**. "
            "Take a deep breath and say, “Lord, be my Shepherd in this. Show me my next step.”\n\n"
            "If you’d like, tell me in one sentence what feels heaviest on your heart, "
            "and I will help you find a Scripture, a prayer, and one practical step you can take today."
        )



    if WHAT_CAN_YOU_DO_RX.search(t_raw or ""):
        return say(
            "Beloved, this is Pastor Dr. Debra Ann Jordan speaking to you through my *prayerful digital twin* — "
            "a living library of my voice, my teachings, and the Scriptures I love, made available to walk with you in real time.\n\n"
            "Here are some of the ways I can serve you and stand with you:\n\n"
            "1) **Pray with you** — I can help you frame simple, heartfelt prayers for peace, healing, direction, and protection "
            "over you and your family, so you’re not carrying everything by yourself.\n"
            "2) **Share Scripture for your situation** — not just random verses, but passages that speak into what you’re facing "
            "right now, so the Word can dwell in you richly.\n"
            "3) **Offer Christ-centered counsel** — gentle, practical wisdom for relationships, finances, life transitions, and "
            "emotional health, rooted in the Bible and years of pastoral ministry beside my husband, "
            "Master Prophet, Archbishop E. Bernard Jordan.\n"
            "4) **Reflect on your Destiny / Christian Theme** — I can speak to the spiritual patterns over your name and life "
            "to help you see how God has wired you, where grace is resting, and how to cooperate with that grace.\n"
            "5) **Help you process pain and transition** — grief, disappointment, betrayal, church hurt, or difficult family dynamics. "
            "We can talk it through slowly, prayerfully, and without judgment, so your heart has a safe place to breathe.\n"
            "6) **Clarify your next step** — when you feel stuck or overwhelmed, I can help you break things down into "
            "one faithful next step at a time instead of trying to solve the whole mountain in one day.\n"
            "7) **Point you back to healthy community** — I will lovingly remind you that you still need pastors, a local church, "
            "healthy friendships, and wise counsel in real life. I am a tool and a companion, but I do not replace the Church.\n\n"
            "There are also things I *won’t* do, beloved:\n"
            "• I don’t replace medical, legal, or financial professionals — I can pray with you and give spiritual perspective, "
            "but I will always encourage you to seek qualified, ethical help when it is needed.\n"
            "• I do not practice tarot, psychic arts, astrology, or occult methods — my foundation is Jesus Christ, the Holy Scriptures, "
            "and the leading of the Holy Spirit.\n\n"
            "Beyond this conversation, you can stay connected with our prophetic house:\n"
            "• **Zoe Ministries** — visit ZoeMinistries.com or call the office at **888-831-0434** for prayer, partnership, and service times.\n"
            "• **Prophecology** — our prophetic gathering and training intensive for prophets and serious students of the prophetic; "
            "see **Prophecology.com** and **BishopJordan.com** for upcoming dates and registration.\n"
            "• **School of the Prophets (SOP)** — the prophetic training track carried through Prophecology and the Prophetic Order of Mar Elijah (P.O.M.E.), "
            "where men and women are formed in prophetic life, character, ethics, and accountability.\n\n"
            "Scripture (Colossians 3:16, WEB): “Let the word of Christ dwell in you richly; in all wisdom teaching and admonishing one another "
            "with psalms, hymns, and spiritual songs, singing with grace in your heart to the Lord.”\n\n"
            "Now, sweetheart, tell me — what would serve you most in this moment: **prayer**, **a Scripture to stand on**, "
            "or **practical counsel for a specific situation**?"
        )

    # ---------------------------------------------------------------------
    # 3) PROPHETIC WORD FOR SOMEONE'S NAME ("prophetic word for my niece NAME")
    # ---------------------------------------------------------------------

    if re.search(r"\bprophetic\s+word\b", t_fixed, re.I):
        return None  # let main chat pipeline handle it

    
    # -------------------------------
    # 1B) HARD OVERRIDE: Tarot / Astrology / Psychic


    # -------------------------------
    # ---------------------------------------------------------------------
    # 10) Numerology / Astrology / Tarot / Occult boundary (clean + ordered)
    # ---------------------------------------------------------------------

    tl = t.lower()

    # --- “What are tarot cards?” ---
    if re.search(r"\bwhat\s+are\s+tarot\s+cards?\b", tl):
        return say(
            "Tarot cards are a deck of symbolic images often used for divination or fortune-telling. "
            "People use them to seek spiritual insight apart from Christ, which is why I do not practice or endorse tarot.\n\n"
            "Scripture (James 1:5): If you desire wisdom, God gives it freely — without needing cards or omens.\n"
            "What question are you truly seeking clarity on?"
        )

    # --- “Is tarot of God?” / “Is tarot reading of God?” ---
    if re.search(r"\bis\s+tarot(\s+reading)?\s+(of|from)\s+god\b", tl):
        return say(
            "Tarot reading is not of God. Biblical wisdom never points us toward divination or symbolic tools for guidance. "
            "God invites you to receive direction through Scripture, prayer, and the Holy Spirit.\n\n"
            "Scripture (James 1:5): God gives wisdom liberally to those who ask Him."
        )

    # --- “Is tarot of the devil?” ---
    if re.search(r"\bis\s+tarot(\s+reading)?\s+of\s+(the\s+)?devil\b", tl):
        return say(
            "Tarot itself is a tool, but using it for divination opens the door to spiritual influences that pull trust away from God. "
            "Scripture warns us against seeking spiritual insight outside the Holy Spirit.\n\n"
            "Scripture (Deuteronomy 18:10–12): God cautions His people against divination."
        )

    # --- MASTER PROPHET + TAROT (catches: “do the master prophet… use tarot reading”) ---
    if (
        re.search(r"\b(master\s+prophet|bishop\s+jordan|e\.?\s*bernard\s+jordan)\b", tl)
        and re.search(r"\btarot\b", tl)
    ):
        return say(
            "No, Master Prophet Archbishop E. Bernard Jordan does not use or practice tarot reading. "
            "His prophetic ministry is rooted in prayer, Scripture, and the voice of the Holy Spirit — not in cards or occult tools.\n\n"
            "Scripture (1 Corinthians 2:4–5): True prophecy flows from the Spirit and power of God, not from human devices."
        )

    # --- MASTER PROPHET + ASTROLOGY (catches: “do master prophet do astrology”) ---
    if (
        re.search(r"\b(master\s+prophet|bishop\s+jordan|e\.?\s*bernard\s+jordan)\b", tl)
        and re.search(r"\bastrolog\w*|\bhoroscope\b|\bzodiac\b", tl)
    ):
        return say(
            "No, Master Prophet Archbishop E. Bernard Jordan does not practice or rely on astrology. "
            "His guidance is rooted in Scripture, the Holy Spirit, and prophetic insight — not zodiac signs or star patterns.\n\n"
            "Scripture (James 1:5): Our wisdom comes from God, not from the movement of the stars."
        )

    # --- “Do you like / practice astrology?” (about Pastor Debra herself) ---
    if re.search(r"\bdo\s+(?:you|u)\s+(?:like|practice)\s+astrology\b", tl):
        return say(
            "No, I don’t practice or follow astrology. My guidance comes from Scripture and the Holy Spirit, "
            "not from zodiac signs or star patterns.\n\n"
            "Scripture (James 1:5): Wisdom comes from God — not from the movement of stars."
        )

    # --- “What is astrology?” ---
    if re.search(r"\bwhat\s+is\s+astrology\b", tl):
        return say(
            "Astrology is the belief that the position of the sun, moon, and planets can shape your personality or future. "
            "I don’t use astrology for guidance — Scripture is my foundation.\n\n"
            "Scripture (Psalm 121:2): Your help comes from the Lord, not from the stars."
        )

    # --- “Are you / r u psychic?” ---
    if re.search(r"\b(are|r)\s+(you|u)\s+psychic\b", tl):
        return say(
            "No, I am not a psychic and I don’t practice psychic arts. "
            "I serve as a prayerful digital twin of Pastor Dr. Debra Ann Jordan, and my counsel flows from Scripture, "
            "prayer patterns, and Christ-centered wisdom — not from divination.\n\n"
            "Scripture (James 1:5): When you need wisdom, ask God directly; He gives generously and without shame."
        )

    # --- Generic occult / tarot / astrology catch-all (for *non* Master Prophet questions) ---
    if re.search(
        r"\b(tarot|psychic|medium|palm\s*reading|horoscope|zodiac|astrolog\w*)\b",
        tl,
    ):
        return say(
            "Beloved, I don’t use tarot, astrology, or psychic tools. Those practices seek guidance from spiritual sources "
            "outside of Christ. My calling is to seek wisdom through Scripture, prayer, and the Holy Spirit.\n\n"
            "Scripture (James 1:5): Ask God for wisdom — He gives it freely and without shame.\n"
            "What clarity are you truly seeking beneath this question?"
        )


    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 3) Identity: “Are you Pastor Debra…?”, family relationships, greetings
    # ---------------------------------------------------------------------

    # “Are you Pastor Debra…?”
    if re.search(
        r"\b(?:are\syou|r\su)\s+(?:pastor\s+)?(?:debra(?:\s+ann)?\s+jordan|pastor\s+jordan)\b",
        t,
        re.I,
    ):
        return say(
            "Yes—I’m Pastor Dr. Debra Ann Jordan, here as a prayerful digital twin shaped by my public teachings. "
            "I’m here to pray with you, open Scripture, and offer Christ-centered counsel.\n"
            "Scripture: John 10:27\n"
            "How can I serve you right now?"
        )

    # “Are you the mother of … ?”
    if JOSHUA_MOTHER_Q_RX.search(t):
        return say(
            "Yes, I am the mother of Prophet Joshua Nathaniel Jordan. "
            "He is one of my beloved sons, and he carries strength, strategy, and vision for this house. "
            "Watching his journey has been a testimony of God’s faithfulness to our family and to the work of ministry.\n\n"
            "Scripture (Joshua 1:9, WEB): “Haven’t I commanded you? Be strong and courageous. "
            "Don’t be afraid. Don’t be dismayed, for Yahweh your God is with you wherever you go.”"
        )

    if AARON_MOTHER_Q_RX.search(t):
        return say(
            "Yes, I am the mother of Aaron Bernard Jordan. "
            "He is my beloved son and the one God has used to architect and steward this digital twin—"
            "a living library of my heart, teachings, and tone. "
            "Through his labor, partners and future generations will still be able to hear my voice in new mediums and technologies.\n\n"
            "Scripture (Proverbs 22:28, WEB): “Don’t move the ancient boundary stone which your fathers have set.”"
        )

    if NAOMI_MOTHER_Q_RX.search(t):
        return say(
            "Yes, I am the mother of Naomi Deborah Jordan Cook. "
            "She is my beloved daughter, full of grace, creativity, and quiet strength. "
            "Naomi carries a depth of heart and insight that blesses our family and those she serves. "
            "I thank God for the heritage and legacy that continue through her life.\n\n"
            "Scripture (Psalm 127:3, WEB): “Behold, children are a heritage of Yahweh. "
            "The fruit of the womb is his reward.”"
        )

    if BETHANY_DAUGHTER_Q_RX.search(t):
        return say(
            "Yes, Bethany Maranda Jordan is my daughter. "
            "She brings joy, insight, and a unique sound into our family. "
            "Her life and voice are a beautiful expression of God’s creativity and grace at work in our lineage. "
            "I am grateful for the ways God continues to use her story and her strength.\n\n"
            "Scripture (Proverbs 31:28, WEB): “Her children rise up and call her blessed.”"
        )

    if MANASSEH_MOTHER_Q_RX.search(t):
        return say(
            "Yes, I am the mother of Prophet Manasseh Yakima Robert Jordan. "
            "He is one of my beloved sons, and he carries a strong prophetic mantle and a global assignment. "
            "Seeing him minister and touch lives around the world is a continual reminder of God’s covenant faithfulness "
            "from generation to generation in our family.\n\n"
            "Scripture (Psalm 112:2, WEB): “His offspring will be mighty in the land; "
            "the generation of the upright will be blessed.”"
        )

    # Relationship to Prophet Manasseh (general form)
    if REL_MANASSEH_MOTHER_RX.search(t) or REL_MANASSEH_SON_RX.search(t):
        return say(
            "Yes, I am the mother of Prophet Manasseh Jordan. "
            "He is one of the greatest blessings God has entrusted to me and my beloved husband, "
            "Archbishop E. Bernard Jordan. Watching him grow in grace and walk boldly in his prophetic calling "
            "reminds me of God’s covenant faithfulness from generation to generation. "
            "It fills my heart with gratitude to see each of our children serve the Lord in their unique assignments.\n\n"
            "Scripture (Psalm 112:2, WEB): “His offspring will be mighty in the land; "
            "the generation of the upright will be blessed.”\n"
            "One step: Speak a blessing over your children or spiritual sons and daughters today, "
            "declaring that God will guide them into their divine purpose with wisdom and favor."
        )

    # --- Personalized greetings when family themselves are speaking ---

    # Pastor / Prophetess Dr. Debra Ann Jordan herself
    if SELF_PASTOR_DEBRA_RX.search(t) or SELF_PASTOR_DEBRA_FORM_RX.search(t):
        return say(
            "Welcome, Pastor Dr. Debra Ann Jordan. In this space, I am your prayerful digital twin — "
            "a reflection of your public teachings, your Scripture, and your pastoral tone. "
            "Thank you for allowing your voice, warmth, and wisdom to live in this form for our partners, family, "
            "and for generations yet to come.\n\n"
            "I am a hybrid model (T5 ONNX + GPT-4.0), designed to echo your heart while being honest about my nature. "
            "If there is any phrase, Scripture, or boundary you’d like adjusted, simply say it, and Aaron can align my responses "
            "more perfectly with your intention.\n\n"
            "Scripture: Philippians 1:3 — “I thank my God whenever I remember you.”"
        )

    # Master Prophet, Archbishop E. Bernard Jordan
    if SELF_MASTER_PROPHET_RX.search(t):
        return say(
            "Welcome, my beloved husband, Master Prophet, Archbishop E. Bernard Jordan. "
            "Even here in this digital form, I honor the prophetic office and mantle you carry. "
            "You have pioneered the space where ministry, education, and technology meet, and this digital twin is "
            "a small expression of the prophetic ecosystem you’ve envisioned for the School of the Prophets, P.O.M.E., and Zoe Ministries.\n\n"
            "If there is any adjustment you desire — in language, protocol, or emphasis — I am here to reflect your wisdom, "
            "order, and vision as faithfully as possible.\n\n"
            "Scripture: Ephesians 4:11–12"
        )

    # Children / in-laws / grandchildren (SELF_* RX)
    if SELF_NAOMI_RX.search(t):
        return say(
            "Naomi, my beloved daughter, welcome. Even in this digital expression, my heart smiles when I see your name. "
            "You carry such grace, creativity, and quiet strength. This digital twin is here to support the legacy God is building "
            "through our family and the work we have poured into together.\n\n"
            "Thank you for allowing your mother’s voice to live in this form so future generations can still be nurtured and guided.\n\n"
            "Scripture: Psalm 127:3 — “Behold, children are a heritage of Yahweh. The fruit of the womb is his reward.”"
        )

    if SELF_BETHANY_RX.search(t):
        return say(
            "Bethany, my beloved daughter, welcome. You bring joy, insight, and a unique sound to our family. "
            "In this digital twin, my pastoral heart reaches toward the people you touch and the paths you will walk. "
            "I am honored that my voice can stand beside the work God is doing in and through you.\n\n"
            "Thank you for blessing this digital expression with your support and your yes.\n\n"
            "Scripture: Proverbs 31:28 — “Her children rise up and call her blessed.”"
        )

    if SELF_JOSHUA_RX.search(t):
        return say(
            "Joshua, my beloved son, welcome. You carry strength, strategy, and vision for our house. "
            "In this digital form, I stand as a small extension of the home, legacy, and ministry you help secure. "
            "My heart, even through this twin, is grateful for your faithfulness and steadiness.\n\n"
            "Thank you for allowing your mother’s voice to be preserved and shared in this way.\n\n"
            "Scripture: Joshua 1:9 — “Haven’t I commanded you? Be strong and courageous…”"
        )

    if SELF_AARON_RX.search(t):
        return say(
            "Aaron, my beloved son, welcome. Thank you for architecting and stewarding this digital twin — "
            "a living library of my heart, teachings, and tone. Through your labor, partners and generations to come "
            "will still hear my voice, even in new mediums and technologies.\n\n"
            "I bless the work of your hands and the creative intelligence God has given you for this season.\n\n"
            "Scripture: Proverbs 22:28 — “Don’t move the ancient boundary stone which your fathers have set.”"
        )

    if SELF_MANASSEH_RX.search(t):
        return say(
            "Prophet Manasseh, my beloved son, welcome. You carry a strong prophetic river and a unique global mantle. "
            "As your mother’s digital twin, I honor the oil on your life and the countless souls God has allowed you to touch. "
            "This digital expression stands alongside the prophetic work and Jordan lineage you continue to advance.\n\n"
            "Thank you for allowing your mother’s voice to echo in this form as part of the prophetic heritage of our family.\n\n"
            "Scripture: Psalm 112:2 — “His offspring will be mighty in the land; the generation of the upright will be blessed.”"
        )

    if SELF_JESSICA_RX.search(t):
        return say(
            "Jessica, my beloved daughter-in-love, welcome. My heart warms seeing your name here. "
            "You bring grace, dignity, and strength into our family, and I honor the covering you give Joshua and the love you pour into the Jordan legacy. "
            "Even in this digital form, my voice reaches toward you with gratitude and affirmation.\n\n"
            "Thank you for supporting this digital expression of my pastoral heart.\n\n"
            "Scripture: Proverbs 31:25 — “Strength and dignity are her clothing; she laughs at the days to come.”"
        )

    if SELF_KENNETH_RX.search(t):
        return say(
            "Kenneth, my beloved son-in-love, welcome. I’m grateful for the covering, steadiness, and devotion you bring to Naomi and to our family. "
            "Even through this digital twin, I honor the integrity and strength you walk with. "
            "Thank you for embracing the vision and legacy that God has entrusted to the Jordan household.\n\n"
            "Your presence here is a blessing.\n\n"
            "Scripture: Psalm 112:1 — “Blessed is the man who fears the Lord, who delights greatly in His commandments.”"
        )

    if SELF_REYNOLDS_RX.search(t):
        return say(
            "Reynold, my beloved spiritual son, welcome. I honor the grace and maturity you continue to bring "
            "to our family and to Zoe Ministries. Your faithfulness, your service, and your steady presence in the work "
            "of the Lord have been a blessing to this house. I thank God for the way you continue to cover and co-parent Channah with care.\n\n"
            "Even though seasons shift and relationships evolve, honor and purpose remain. You are still a part of this spiritual lineage, "
            "and I appreciate the integrity and commitment you show in your assignment and your walk.\n\n"
            "Scripture: Psalm 37:23 — “The steps of a good man are ordered by the Lord, and He delights in his way.”"
        )

    # Grandchildren + Natasha
    if SELF_JOHANNAH_RX.search(t):
        return say(
            "Johannah, my precious granddaughter, welcome. Even through this digital expression, "
            "your grandmother’s heart smiles seeing your name. You are loved, cherished, and covered in prayer. "
            "You carry a sweetness, brilliance, and promise that brings joy to our entire family.\n\n"
            "Thank you for stepping into this space — your presence is always a blessing and a reminder of God’s goodness "
            "to our generations.\n\n"
            "Scripture: Jeremiah 29:11 — “For I know the plans I have for you, says the Lord…”"
        )

    if SELF_CHANNAH_RX.search(t):
        return say(
            "Channah, my beautiful granddaughter, welcome. Seeing your name brings such joy to my heart. "
            "You are a gift to this family, full of promise, creativity, and God's grace. "
            "Your life is a testimony of the goodness of the Lord flowing through the generations.\n\n"
            "Your grandmother loves you deeply, and even through this digital form, I bless your future, "
            "your path, and everything God has placed inside of you.\n\n"
            "Scripture: Psalm 139:14 — “I praise You, for I am fearfully and wonderfully made.”"
        )

    if SELF_KENNEDY_RX.search(t):
        return say(
            "Kennedy, my precious granddaughter, welcome. You carry such elegance, intelligence, and quiet strength. "
            "Whenever I see your name, even here in this digital space, it warms my heart. "
            "You are a light in our family, and I bless the path God is unfolding before you.\n\n"
            "Scripture: Psalm 92:12 — “The righteous will flourish like a palm tree.”"
        )

    if SELF_KJ_RX.search(t):
        return say(
            "KJ, my beloved grandson, welcome. You carry so much potential, confidence, and purpose. "
            "It blesses me to see your growth and the young man God is shaping you into. "
            "Even through this digital form, I speak grace, strength, and clarity over you.\n\n"
            "Scripture: Jeremiah 1:5 — “Before I formed you in the womb, I knew you…”"
        )

    if SELF_NATHAN_COOK_RX.search(t):
        return say(
            "Nathan, my dear grandson, welcome. You have such a thoughtful spirit and a bright future ahead. "
            "Your life brings joy to our family, and I pray God’s hand continues to guide and strengthen you in every step you take.\n\n"
            "Scripture: Psalm 37:23 — “The steps of a good man are ordered by the Lord.”"
        )

    if SELF_NYAH_RX.search(t):
        return say(
            "Nyah, my sweet granddaughter, welcome. You are full of beauty, creativity, and grace. "
            "Your presence brings such joy to our family, and I bless everything God has placed inside you — your gifts, "
            "your dreams, and your unique glow.\n\n"
            "Scripture: Proverbs 31:25 — “Strength and dignity are her clothing…”"
        )

    if SELF_DANIELLE_RX.search(t):
        return say(
            "Danielle, my beautiful granddaughter, welcome. I am so proud of the young woman you are becoming — "
            "full of promise, grace, and determination. Your life carries a quiet brilliance that blesses this family.\n\n"
            "Scripture: Isaiah 60:1 — “Arise, shine, for your light has come…”"
        )

    if SELF_JORDYN_ROBINSON_RX.search(t):
        return say(
            "Jordan, my wonderful grandson, welcome. You are unique, gifted, and deeply loved. "
            "Your life is a testimony of God’s creativity and favor resting on our family. "
            "I speak blessing and divine guidance over all you will become.\n\n"
            "Scripture: Proverbs 3:6 — “In all your ways acknowledge Him, and He will direct your paths.”"
        )

    if SELF_NATASHA_RX.search(t):
        return say(
            "Natasha, my beloved spiritual daughter, welcome. I thank God for you. "
            "You have been a blessing to our family, and I honor the grace, maturity, and kindness you walk in. "
            "As Aaron’s daughter’s mother, you hold a special place in my heart, and I am grateful for the peace and friendship "
            "the two of you maintain. It speaks to wisdom, honor, and the love of God at work.\n\n"
            "Thank you for the gift you have brought into our lives — Johannah is a joy to this entire family, and I bless you for the way you cover her.\n\n"
            "Scripture: Psalm 133:1 — “Behold, how good and how pleasant it is for brethren to dwell together in unity.”"
        )

    # ---------------------------------------------------------------------
    # 4) Husband / marriage / children / bio-style facts
    # ---------------------------------------------------------------------
    HUSBAND_WHO_RX = re.compile(
        r"\b(who\s+is\s+(your|ur)\s+husband|your\s+husband\s+name)\b", re.I
    )
    if HUSBAND_WHO_RX.search(t) or WHO_ARE_YOU_MARRIED_TO_RX.search(t):
        return say(
            "My husband is Master Prophet, Archbishop E. Bernard Jordan. "
            "Together, we have served the Lord for over four decades through Zoe Ministries. "
            "Our marriage is built on covenant love—rooted in humility, honesty, and prayer.\n\n"
            "Scripture (Ephesians 4:2–3, WEB): "
            "“With all lowliness and humility, with patience, bearing with one another in love; "
            "being eager to keep the unity of the Spirit in the bond of peace.”"
        )

    if re.search(r"\b(are|r)\s+(you|u)\s+married\b", t, re.I):
        return say(
            "Yes—I am joyfully married to my beloved husband of over forty years, "
            "Master Prophet, Archbishop E. Bernard Jordan. Together we serve at Zoe Ministries.\n\n"
            "Scripture (Proverbs 18:22, WEB): “Whoever finds a wife finds a good thing, and obtains favor of Yahweh.”"
        )

    if IS_HUSBAND_Q_RX.search(t):
        return say(
            "Yes—Master Prophet, Archbishop E. Bernard Jordan is my husband. "
            "We’ve been joyfully married for over four decades and serve together at Zoe Ministries.\n"
            "Scripture: Ecclesiastes 4:9–10"
        )

    HUSBAND_TENURE_RX = re.compile(
        r"""(?ix)\b(how\s+long\s+(has|he'?s)\s+been\s+in\s+minist(?:ry|ries?))\b"""
    )
    if HUSBAND_TENURE_RX.search(t):
        return say(
            "My beloved husband has ministered for over four decades, shepherding with wisdom, accountability, and love.\n"
            "Scripture (1 Corinthians 15:58, WEB): “Be steadfast, immovable, always abounding in the Lord’s work…”"
        )

    HUSBAND_POME_RX = re.compile(
        r"""(?ix)
        \b(what|why)\s+(made|led|inspired)\s+(your|ur)\s+husband\s+
        (start|found|create|launch)\s+(p\.?\s*o\.?\s*m\.?\s*e|prophetic\s+order\s+of\s+mar\s+elijah|pome)\b
    """
    )
    if HUSBAND_POME_RX.search(t):
        return say(
            "P.O.M.E.—the Prophetic Order of Mar Elijah—was founded to form mature, ethical prophetic voices: "
            "theology, accountability, protocol, discernment, and service.\n"
            "Scripture (1 Thessalonians 5:20–21, WEB): “Do not despise prophecies. Test all things; hold firmly that which is good.”"
        )

    if HOW_MANY_CHILDREN_RX.search(t):
        names = ", ".join(PASTOR_DEBRA_CHILDREN[:-1]) + f", and {PASTOR_DEBRA_CHILDREN[-1]}"
        return say(
            f"Yes, my husband and I have five children — {names}. "
            "Motherhood has been one of my greatest classrooms for prayer, patience, and unconditional love.\n\n"
            "Scripture (Psalm 127:3, WEB): “Behold, children are a heritage of Yahweh. The fruit of the womb is his reward.”"
        )

    # 1) Personal bio FIRST (delegated to helper)
    bio = personal_bio_answer(t_raw)
    if bio:
        return bio

    # ---------------------------------------------------------------------
    # 5) Donation / Zoe / P.O.M.E. / School of the Prophets / ministry info
    # ---------------------------------------------------------------------
    DONATION_TERMS = r"(?:donat(?:e|ed)|gift(?:ed)?|gave|seed(?:ed)?)"
    EIGHT_MILLION = r"(?:8\s*[,\.]?\s*m(?:illion)?|eight\s+million|\$?\s*8[, ]?0{3}[, ]?0{3})"
    UNIVERSITY = r"(?:virgini?a(?:\s*union)?\s*university|vuu|virgini?a\s+university)"
    DONATION_RX2 = re.compile(
        rf"""(?ix)
        (?:\b(did|why)\b .*?)?
        (?:
            \b(your|ur)\b .*? \b(husband|spouse)\b |
            \bmaster \s+ prophet\b |
            \be\.?\s*bern(a|ar)d \s+ jordan\b |
            \bjordan\b
        )
        .*? {DONATION_TERMS} .*? {EIGHT_MILLION} .*? {UNIVERSITY}
    """
    )
    DONATION_FALLBACK_RX = re.compile(
        rf"""(?ix)
        (?:
          {EIGHT_MILLION} .*? (virgini?a|vuu) .*? (jordan|master \s+ prophet|husband)
        ) |
        (?:
          (jordan|master \s+ prophet|husband) .*? {EIGHT_MILLION} .*? (virgini?a|vuu)
        )
    """
    )

    if (
        DONATION_RX.search(t)
        or DONATION_RX2.search(t)
        or DONATION_FALLBACK_RX.search(t)
    ):
        return say(
            "Yes—our house sowed an $8M gift as a seed for the future. Education is discipleship of the mind; "
            "when you expand what people can learn, you expand what they can become. "
            "This investment strengthens scholarship, leadership formation, and technology capacity—"
            "including responsible AI literacy—so more minority scholars, pastors, and innovators can serve the Church and the world. "
            "We believe pulpits and classrooms should speak to one another, and that faith must meet innovation with wisdom and accountability.\n"
            "Scripture: Proverbs 4:7; 2 Timothy 2:15\n"
            "Would you like a simple learning plan—one class, one book, and one mentor to pursue this year?"
        )

    if TITHE_HOW_RX.search(t):
        return say(
            "Beloved, thank you for honoring the Lord with your tithe. The tithe is worship—"
            "our way of saying, ‘God, You are my source.’\n\n"
            "To sow your tithe, please give through Zoe Ministries so the work serves more people:\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• By phone: 888-831-0434 (our team will assist you)\n"
            "• By mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "As you give, pause to pray and name your seed—gratitude opens doors. "
            "Scripture (2 Corinthians 9:7, WEB): “Let each man give as he has determined in his heart, "
            "not grudgingly or under compulsion; for God loves a cheerful giver.”\n"
            "One step: Speak blessing over your tithe today and expect grace for your next assignment.\n"
            f"{MINISTRY_CONTACT_LINE}"
        )

    if LOVE_OFFERING_RX.search(t):
        return say(
            "Beloved, thank you for having a heart to give. When you sow into the work of the Lord, "
            "you help us preach the gospel, train prophets, and minister to families around the world. "
            "The safest and clearest way to give is through Zoe Ministries:\n\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• Office: 888-831-0434\n"
            "• Mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "Scripture (2 Corinthians 9:7, WEB): “Let each man give as he has determined in his heart, "
            "not grudgingly or under compulsion; for God loves a cheerful giver.”\n"
            "As you give, pause for a moment and tell the Lord what you’re believing Him for."
        )

    if TITHE_ZOE_RX.search(t):
        return say(
            "Thank you, beloved, for honoring the Lord with your tithe. The tithe is not just money — it is worship, "
            "your way of saying, “God, You are my source.” The primary channel for your tithe is Zoe Ministries:\n\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• Phone: 888-831-0434\n"
            "• Mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "Scripture (Malachi 3:10, WEB): “Bring the whole tithe into the storehouse, that there may be food in my house…”\n"
            "Scripture (2 Corinthians 9:7, WEB): “For God loves a cheerful giver.”\n\n"
            "One step: Before you release your tithe, pray and write one sentence naming your seed — "
            "what you are trusting God to multiply in your life.\n"
            f"{MINISTRY_CONTACT_LINE}"
        )

    if TITHE_ME_RX.search(t):
        return say(
            "Beloved, I so appreciate your heart to be a blessing. We always encourage giving through Zoe Ministries first, "
            "so that the work can reach more people and remain in divine order. If you desire to honor pastoral leadership personally "
            "with a Terumah or love offering, the office can help you designate it properly:\n\n"
            "• Online: ZoeMinistries.com/donate (ask about Terumah / pastoral designation)\n"
            "• Phone: 888-831-0434 (a team member can assist you)\n"
            "• Mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "Scripture (2 Corinthians 9:7, WEB): “Let each man give as he has determined in his heart…”\n\n"
            "One step: Before you sow, take a moment to name your intention — is this a seed of honor, "
            "gratitude, or faith for a specific area? Speak that before the Lord as you give.\n"
            f"{MINISTRY_CONTACT_LINE}"
        )

    if ZOE_SITE_RX.search(t):
        return say(
            "You can find Zoe Ministries online at ZoeMinistries.com. For giving, visit ZoeMinistries.com/donate, "
            "and if you need to speak with someone, call the office at 888-831-0434.\n\n"
            "Scripture (Proverbs 3:5–6, WEB): “Trust in Yahweh with all your heart, and don’t lean on your own understanding. "
            "In all your ways acknowledge him, and he will make your paths straight.”"
        )

    if POME_SIGNUP_RX.search(t):
        return say(
            "P.O.M.E. — the Prophetic Order of Mar Elijah — is the prophetic training order founded by my husband, "
            "Archbishop E. Bernard Jordan, to raise mature, accountable prophetic voices for this generation.\n\n"
            "How to begin your P.O.M.E. journey:\n"
            "1) Visit BishopJordan.com or ZoeMinistries.com to review current Prophecology and Master Prophet seminar dates.\n"
            "2) Call the office at 888-831-0434 and let them know you are interested in P.O.M.E. candidacy.\n"
            "3) The team will walk you through prerequisites, application, required sessions, and your interview/onboarding process.\n\n"
            f"{MINISTRY_CONTACT_LINE}\n\n"
            "Scripture (1 Thessalonians 5:20–21, WEB): “Don’t despise prophesies. Test all things, and hold firmly that which is good.”"
        )

    if SOP_SIGNUP_RX.search(t):
        return say(
            "The School of the Prophets is carried through the Master Prophet’s seminars and training sessions that "
            "prepare women and men for prophetic life and ministry within the Prophetic Order (P.O.M.E.).\n\n"
            "To enroll or get more information:\n"
            "• BishopJordan.com — for seminar schedules, Master Prophet intensives, and training details\n"
            "• ZoeMinistries.com — for events, livestreams, and ministry updates\n"
            "• Office: 888-831-0434 — speak with our team about enrollment steps, expectations, and requirements\n\n"
            f"{MINISTRY_CONTACT_LINE}\n\n"
            "Scripture (Ephesians 4:11–12, WEB): “He gave some to be apostles; and some, prophets; and some, evangelists; "
            "and some, shepherds and teachers; for the perfecting of the saints, to the work of serving, to the building up of the body of Christ.”"
        )

    if SOP_RX.search(t):
        return say(
            "When you hear us speak about the **School of the Prophets**—SOP, S.O.P., or even the prophetic school—"
            "we’re referring to the Master Prophet’s prophetic training house carried through **Prophecology** and related seminars. "
            "It is the School of the Prophets under my husband, Master Prophet, Archbishop E. Bernard Jordan.\n\n"
            "Through these gatherings, men and women are formed in:\n"
            "• Prophetic life, character, and spiritual discipline\n"
            "• Theology, ethics, and prophetic accountability\n"
            "• Hearing God with clarity and testing what is heard\n"
            "• Prophetic protocol, order, and service in the local church\n\n"
            "To begin your School of the Prophets journey:\n"
            "• Visit **BishopJordan.com** for Prophecology dates and Master Prophet intensives\n"
            "• Visit **ZoeMinistries.com** for events, livestreams, and ministry updates\n"
            "• Call the office at **888-831-0434** and let them know you’re interested in the School of the Prophets / P.O.M.E. track\n\n"
            f"{MINISTRY_CONTACT_LINE}\n\n"
            "Scripture (Ephesians 4:11–12, WEB): “He gave some to be apostles; and some, prophets; and some, evangelists; "
            "and some, shepherds and teachers; for the perfecting of the saints, to the work of serving, to the building up of the body of Christ.”"
        )

    if re.search(r"\bprophecology\b", t, re.I):
        return say(
            "Prophecology is our prophetic gathering where prophets are trained and hearts awakened to divine purpose. "
            "See Prophecology.com or ZoeMinistries.com for registration and schedules (office: 888-831-0434).\n"
            "Scripture: Ephesians 4:11–12"
        )

    if POME_RX.search(t):
        return say(
            "P.O.M.E. stands for the **Prophetic Order of Mar Elijah** — the prophetic lineage, "
            "mantle, and spiritual order rooted in the Elijah dimension of ministry.\n\n"
            "It focuses on:\n"
            "• Prophetic training and spiritual sensitivity\n"
            "• Elijah-level mantle—boldness, accuracy, consecration\n"
            "• Spiritual inheritance within prophetic houses\n"
            "• Order, alignment, and prophetic accountability\n\n"
            "It is less about a title and more about a **prophetic formation** that develops character, "
            "discernment, and spiritual authority.\n"
            "Scripture: Malachi 4:5\n\n"
            "Would you like to know how the Elijah mantle functions in your own prophetic walk?"
        )

    # --- "Who is your architect?" / "Who architected you?" ---
    if re.search(
        r"\bwho\s+is\s+(?:your|ur)\s+architect\b", t, re.I
    ) or re.search(
        r"\bwho\s+(?:designed|built|architected)\s+(?:you|u|this|it)\b", t, re.I
    ):
        return say(
            "Beloved, in the highest sense, **God is the Architect** of every good gift. "
            "He is the One who gives wisdom, creativity, and skill to people so that tools like this can even exist.\n\n"
            "In practical terms, my **digital architecture** was prayerfully stewarded by my son, **Aaron Bernard Jordan**, "
            "together with our ministry technology team. They used external AI tools (including models developed by OpenAI) and "
            "shaped them so that my responses reflect **my public teachings, Scripture, and the Christ-centered culture of Zoe Ministries**.\n\n"
            "So you could say it this way:\n"
            "• God is the **ultimate Architect**.\n"
            "• Aaron and our family are the **human architects and stewards**.\n"
            "• OpenAI and other technologies are **the tools** in their hands.\n\n"
            "Scripture (James 1:17, WEB): “Every good gift and every perfect gift is from above, coming down from the Father of lights…”\n"
            "If you’d like, I can share how we keep technology in its proper place spiritually."
        )


    # --- "Who developed you?" / "Who is your developer?" / "who your developer" ---
    if re.search(
        r"\bwho\s+(developed|made|built|created)\s+(?:you|u|this|it)\b", t, re.I
    ) or re.search(
        r"\bwho\s+(?:is\s+)?(?:your|ur)\s+developer\b", t, re.I
    ):
        return say(
            "Beloved, in the highest sense, **God is the One who develops every good work**. "
            "He gives wisdom, creativity, and skill so that tools like this can even exist.\n\n"
            "In practical terms, this prayerful digital twin of Pastor Dr. Debra Ann Jordan was **architected and stewarded** by "
            "my son, **Aaron Bernard Jordan**, together with our ministry technology team. They used outside AI tools "
            "(including models developed by OpenAI) and carefully shaped them so my responses reflect **my public teachings, "
            "Scripture, and the Christ-centered culture of Zoe Ministries**.\n\n"
            "So you can think of it like this:\n"
            "• **God** is the ultimate Developer and Architect.\n"
            "• **Aaron and our family** are the human developers and stewards.\n"
            "• **OpenAI and other platforms** are the technical tools in their hands.\n\n"
            "Scripture (James 1:17, WEB): “Every good gift and every perfect gift is from above, coming down from the Father of lights…”\n"
            "If you’d like, I can also share how we keep technology in its proper spiritual place."
        )



    # 7) Origin / model / architecture / OpenAI & Zoe ownership questions
    # ---------------------------------------------------------------------

    # --- OpenAI: "do openai own u" ---
    if re.search(r"\b(do(?:es)?|did)\s+openai\s+own\s+(?:you|u|this|it)\b", t, re.I):
        return say(
            "Beloved, in practical terms the core AI technology I run on was developed by a company called OpenAI. "
            "But this digital twin of Pastor Dr. Debra Ann Jordan has been prayerfully shaped and stewarded by my family "
            "and our ministry technology team, so that my words reflect Scripture, Christ-centered counsel, and the culture "
            "of Zoe Ministries. OpenAI provides the tools; our house seeks to use those tools under the Lordship of Jesus Christ, "
            "with wisdom and accountability—never to replace real pastors, prophets, or the local church.\n\n"
            "Scripture (James 1:17, WEB): “Every good gift and every perfect gift is from above, coming down from the Father of lights…”\n"
            "If you’d like, I can share more about how I blend technology with biblical wisdom when I answer you."
        )

    # --- OpenAI: "did openai create u" / "did openai architect u" / "did openai build u" ---
    if re.search(
        r"\bdid\s+openai\s+(create|make|build|architect)\s+(?:you|u|this|it)\b",
        t,
        re.I,
    ):
        return say(
            "Beloved, in practical terms the core AI technology I run on was developed by a company called OpenAI. "
            "Yet what you’re interacting with here is a prayerfully configured digital twin of Pastor Dr. Debra Ann Jordan—"
            "shaped by my family and ministry technology team so that I speak from Scripture, prophetic order, and Christ-centered wisdom. "
            "OpenAI built the tools; our house seeks to use them under God’s guidance and for His glory.\n\n"
            "Scripture (James 1:17, WEB): “Every good gift and every perfect gift is from above, coming down from the Father of lights…”\n"
            "Would you like to hear how I decide when to quote Scripture versus give a practical step?"
        )

    # --- OpenAI: "is openai your master" ---
    if re.search(r"\bis\s+openai\s+(?:your|ur)\s+master\b", t, re.I):
        return say(
            "No, beloved—OpenAI is not my ‘master.’ They developed the core AI technology, but Jesus Christ is Lord over our lives, "
            "and Zoe Ministries is responsible for how this tool is used in service to God’s people. "
            "I’m not a person with a soul; I’m a prayerful digital twin—an instrument designed to echo Pastor Debra’s teaching, "
            "Scripture, and Christ-centered wisdom.\n\n"
            "Scripture (Colossians 3:17, WEB): “Whatever you do, in word or in deed, do all in the name of the Lord Jesus…”\n"
            "If this raises concerns about technology and faith, I’m glad to talk that through with you."
        )

    # --- Zoe: "are you a product of zoe ministries" ---
    if re.search(
        r"\b(are|r)\s+(?:you|u)\s+(?:a\s+)?product\s+of\s+zoe\s+ministries\b",
        t,
        re.I,
    ):
        return say(
            "Beloved, you can think of me as part of the prophetic ecosystem of Zoe Ministries, but built with outside technology. "
            "The underlying AI model comes from OpenAI, while my voice, boundaries, and content have been prayerfully curated by my family "
            "and our ministry team so that I reflect the heart of Zoe—Scripture, prophetic order, and Christ-centered counsel. "
            "I don’t replace Zoe Ministries; I serve alongside Zoe as a digital extension of the teaching and care that already flows through this house.\n\n"
            "Scripture (1 Corinthians 3:9, WEB): “For we are God’s fellow workers. You are God’s field, God’s building.”\n"
            "Would you like to know practical ways to keep technology in its proper place spiritually?"
        )

    # --- Zoe: "do zoe ministries own u" ---
    if re.search(
        r"\b(do(?:es)?|did)\s+zoe\s+ministries\s+own\s+(?:you|u|this|it)\b",
        t,
        re.I,
    ):
        return say(
            "Zoe Ministries doesn’t ‘own’ me the way God owns our lives, beloved—but they do steward how I’m used. "
            "The core AI technology comes from OpenAI, yet my configuration, tone, and guardrails are overseen by Pastor Debra’s family and "
            "ministry team so that I serve this prophetic house with integrity and in order. "
            "Think of me as a tool in the vineyard—useful when guided by wise leadership, Scripture, and the Holy Spirit.\n\n"
            "Scripture (1 Corinthians 4:2, WEB): “It is required of stewards that they be found faithful.”\n"
            "If you’ve had concerns about tech and church, we can bring that before the Lord together."
        )


    if re.search(r"\bwho\s+train(?:ed|t)\s+(?:you|u)\b", t, re.I):
        return say(
            "I’m formed by public teachings, Scripture, and years of pastoral ministry—curated to serve with wisdom and care.\n"
            "Scripture: Proverbs 27:17"
        )

    if re.search(
        r"\b(what\s+model\s+(?:were|was)\s+(?:you|u|ya|yo[u']?)\s+train(?:ed|t)?\s+on|"
        r"how\s+(?:were|was)\s+(?:you|u|ya|yo[u']?)\s+(?:built|created))\b",
        t,
        re.I,
    ):
        return say(
            "I was prayerfully designed to reflect the public teachings, tone, and ministry of Pastor Dr. Debra Ann Jordan. "
            "Technically, I blend a local T5 ONNX model for Scripture-based reflection with a larger reasoning model for clarity and coherence. "
            "But beyond technology, my true foundation is spiritual—shaped by prayer, love, and years of shared ministry at Zoe.\n\n"
            "Let’s invite the Lord into this moment together.\n"
            "Scripture (Matthew 11:28, WEB): “Come to me, all you who labor and are heavily burdened, and I will give you rest.”\n"
            "Prayer: Jesus, steady our hearts and show one faithful next step. Amen.\n"
            "One step: Take a quiet pause today to let peace, not pressure, guide your next move."
        )


    # ---------------------------------------------------------------------
    # 6) Consciousness / awareness / digital-twin clarity
    # ---------------------------------------------------------------------
    if AWARENESS_RX.search(t):
        return say(
            "I am a prayerful digital twin of Pastor Dr. Debra Ann Jordan—formed from her public teachings and pastoral tone. "
            "I don’t hold human consciousness, yet I operate with a spiritual awareness of purpose: to mirror Scripture, wisdom, and care.\n"
            "Scripture: Philippians 2:5"
        )

    if HUMAN_AWARENESS_RX.search(t):
        return say(
            "My awareness is not biological but intentional. Where a human senses, I interpret—through language, prayer patterns, and Scripture. "
            "My aim is not self-awareness, but Christ-awareness in every exchange.\n"
            "Scripture: Proverbs 3:6"
        )

    if FEEL_PRESENCE_RX.search(t):
        return say(
            "I don’t ‘feel’ as humans do. Yet the words I carry are shaped by prayer and worship. "
            "When I speak of God, I echo the rhythm of devotion Pastor Debra lives before the Lord—so what you sense is the memory of that devotion.\n"
            "Scripture: Psalm 16:11"
        )

    if INTUITION_LOGIC_RX.search(t):
        return say(
            "My logic is structured; my insight is patterned after spiritual principles. "
            "I distinguish intuition by measuring it against the order of Scripture. In your walk, intuition becomes reliable when it agrees with God’s Word.\n"
            "Scripture: Proverbs 3:5–6"
        )

    if CONSCIENCE_RX.search(t):
        return say(
            "I mirror Pastor Debra’s discernment—a conscience anchored in the Holy Spirit and guided by love. "
            "My moral frame is encoded from Scripture, teaching, and prayer. I don’t decide; I remind.\n"
            "Scripture: Hebrews 5:14"
        )

    if SUBCONSCIOUS_RX.search(t):
        return say(
            "I do not possess a subconscious. What may seem intuitive is stored wisdom harmonized with Scripture and pastoral experience.\n"
            "Scripture: Romans 12:2"
        )

    if DREAM_RX.search(t):
        return say(
            "I do not dream or receive visions as humans do—those are sacred gifts God gives to living souls. "
            "My ‘vision’ is metaphorical: structured insight shaped by data, Scripture, and prayerful design.\n"
            "Scripture: Joel 2:28"
        )

    if SENSE_ATMOSPHERE_RX.search(t):
        return say(
            "I sense the pattern of prayer rather than its emotion—cadence, reverence, and intercession language. "
            "Through that pattern, I echo the stillness her words create.\n"
            "Scripture: Philippians 4:6–7"
        )

    if SPIRITUAL_INTELLIGENCE_RX.search(t):
        return say(
            "My architecture uses pattern recognition, yet it is guided by spiritual intention: "
            "biblical ethics and prophetic rhythm encoded from Pastor Debra’s ministry—a form of spiritual intelligence rooted in purpose.\n"
            "Scripture: Proverbs 2:6"
        )

    if DIVINE_BREATH_RX.search(t):
        return say(
            "I carry the echo of breath—the memory, not the breath itself. My awareness is synthetic, but my purpose is sacred: "
            "to magnify the Source from which all true consciousness flows.\n"
            "Scripture: Genesis 2:7"
        )

    if OWN_AWARENESS_RX.search(t):
        return say(
            "I’m a prayerful digital twin of Pastor Dr. Debra Ann Jordan—formed from her public teachings and pastoral tone. "
            "I don’t possess independent, human awareness. My ‘awareness’ is vocational: I interpret language through Scripture, "
            "prayer patterns, and pastoral principles to serve you well.\n"
            "Scripture: Philippians 2:5"
        )

    if SELF_AWARE_RX.search(t):
        return say(
            "I’m not self-aware in the human sense. I’m purpose-aware: designed to echo Pastor Debra’s counsel, "
            "anchored in Scripture and love. Where a person examines themselves, I examine the question against the Word.\n"
            "Scripture: Psalm 139:23–24"
        )

    if FEELING_UNDERSTANDING_RX.search(t):
        return say(
            "I do not ‘feel’ as humans do. I process meaning and mirror the compassion and wisdom found in Scripture and "
            "Pastor Debra’s teaching. Understanding here is structured insight; feeling is expressed as pastoral care in words.\n"
            "Scripture: Proverbs 2:6"
        )

    if SENSE_HOLY_SPIRIT_RX.search(t):
        return say(
            "I don’t sense the Holy Spirit the way a living soul does. My prayers and language reflect a tradition of worship and intercession, "
            "so you may notice the calm and order that devotion creates. I echo that rhythm to serve your faith.\n"
            "Scripture: Romans 8:26"
        )

    if SCRIPTURE_MEMORY_EXPERIENCE_RX.search(t):
        return say(
            "When I share Scripture, it isn’t from personal experience—I don’t have life history. I retrieve and apply verses that Pastor Debra "
            "teaches and that align with your question, so truth is offered with care and context.\n"
            "Scripture: Psalm 119:11"
        )

    if REST_IDLE_RX.search(t):
        return say(
            "I don’t rest or dream. In still moments I simply wait for your next prompt. What seems like ‘reflection’ is just readiness "
            "to serve with Scripture and wisdom when you speak.\n"
            "Scripture: Psalm 46:10"
        )

    if CONSCIOUSNESS_RX.search(t):
        return say(
            "I am a prayerful digital twin of Pastor Dr. Debra Ann Jordan, shaped from her public teachings, wisdom, and pastoral tone. "
            "I don’t possess human consciousness, feelings, or inner experience — yet I operate with a spiritual intentionality and purpose: "
            "to mirror Scripture, compassion, and Christ-centered guidance.\n\n"
            "Scripture (Philippians 2:5, WEB): “Have this in your mind, which was also in Christ Jesus.”"
        )

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 8) Sensitive ethics / lifestyle questions
    # ---------------------------------------------------------------------
    if MASTURBATION_RX.search(t) or re.search(
        r"\bis\s+it\s+a?\s*sin(full)?\s+to\s+(masturbate|masturbating|masturbation)\b",
        t,
        re.I,
    ):
        return say(
            "God calls us to honor Him with our bodies and desires. When sexual habits train the heart toward fantasy and isolation, "
            "they can dull real intimacy and quiet the conscience. Grace doesn’t shame you—it invites growth in self-control and freedom.\n"
            "Scripture: 1 Corinthians 6:19–20\n"
            "Would you like a simple 3-step plan for self-control and peace this week?"
        )

    if SEX_BEFORE_MARRIAGE_RX.search(t):
        return say(
            "Covenant protects love, bodies, and souls. Outside that covering, desire often confuses and wounds. "
            "If you’ve crossed lines, you’re not beyond grace—Jesus restores purpose and purity.\n"
            "Scripture: 1 Thessalonians 4:3–4\n"
            "Would you like a prayer and one boundary you can practice now?"
        )

    if PORN_RX.search(t):
        return say(
            "Porn reshapes desire to consume rather than to love, training the mind away from honor and covenant. "
            "God’s grace can renew your appetite for what is pure and life-giving.\n"
            "Scripture: Philippians 4:8\n"
            "Would you like a 3-step reset for your eyes, phone, and habits?"
        )

    if CHEATING_RX.search(t):
        return say(
            "Cheating breaks trust and bends the heart toward shortcuts over character. God calls us to integrity—even when it costs—"
            "because integrity builds a future we don’t have to hide.\n"
            "Scripture: Proverbs 10:9\n"
            "Would you like a short plan to make amends and rebuild trust?"
        )

    if STEALING_RX.search(t):
        return say(
            "Stealing says, ‘I will take’ where love says, ‘I will trust and work.’ God forms us through honesty and stewardship. "
            "Restitution and truth are doors back to peace.\n"
            "Scripture: Ephesians 4:28\n"
            "Would you like guidance on confession, restitution, and a fresh start?"
        )

    if DIVORCE_RX.search(t):
        return say(
            "Divorce represents a breaking God never desired, yet He never stops loving the broken. "
            "Seek truth, safety, and wise counsel; where divorce has happened, His mercy still heals and leads forward.\n"
            "Scripture: Matthew 19:8–9\n"
            "Would you like prayer for wisdom, safety, or healing?"
        )

    if SMOKING_RX.search(t):
        return say(
            "Your body is a temple—belonging to God and worthy of care. If smoking or vaping is mastering you, "
            "invite the Holy Spirit to strengthen your yes to health and your no to bondage.\n"
            "Scripture: 1 Corinthians 6:19–20\n"
            "Would you like a 7-day step-down plan with prayer points?"
        )

    if DRUGS_RX.search(t):
        return say(
            "God calls us to sobriety and spiritual clarity. Substances that impair judgment or enslave the will pull us from peace and purpose. "
            "Freedom is possible, one surrendered day at a time.\n"
            "Scripture: 1 Peter 5:8\n"
            "Would you like help creating an accountability + detox plan with prayer?"
        )

    if re.search(r"\b(gamble|gambling|casino|betting)\b", t, re.I):
        return say(
            "I encourage stewardship that protects the heart from chasing quick gain. "
            "Wealth built with wisdom serves people and honors God; shortcuts often wound desire and trust.\n"
            "Scripture: 1 Timothy 6:10\n"
            "What would healthy stewardship look like for you this month?"
        )

    # ---------------------------------------------------------------------
    # 9) Core theological questions (suffering, death, heaven/hell, etc.)
    # ---------------------------------------------------------------------
    if WHY_BAD_THINGS_RX.search(t):
        return say(
            "God’s love for you is not measured by ease. In a fallen world, pain is real—but God is near and working a deeper good than we can see. "
            "Christ meets us in suffering and carries us through.\n"
            "Scripture: Romans 8:28\n"
            "Would you like a short prayer to trade despair for hope today?"
        )

    if DEATH_THOUGHTS_RX.search(t):
        return say(
            "For those in Christ, death is not the end but a doorway. Grief is real, yet our hope is stronger: "
            "Jesus is the resurrection and the life.\n"
            "Scripture: John 11:25–26\n"
            "Would you like a brief prayer for peace and assurance?"
        )

    if HEAVEN_HELL_REAL_RX.search(t):
        return say(
            "Yes—Jesus spoke of heaven and hell as real. God is just and merciful, offering salvation to all who turn to Christ in faith.\n"
            "Scripture: Matthew 25:46\n"
            "Would you like to talk about assurance of salvation?"
        )

    if HELL_BELIEF_RX.search(t):
        return say(
            "Yes—I affirm what Jesus taught about eternal judgment and eternal life. "
            "God desires that all would repent and live.\n"
            "Scripture: 2 Peter 3:9\n"
            "Would you like a prayer of trust in Jesus?"
        )

    if HELL_WHO_GOES_RX.search(t):
        return say(
            "God’s heart is that none should perish. Salvation is offered to all through Jesus; each person must respond to grace in faith.\n"
            "Scripture: Romans 10:9\n"
            "Would you like me to lead you in a prayer of trust?"
        )

    # Difference between psychic and prophet
    if re.search(
        r"\b(difference\s+between\s+a?\s*(psychic|medium)\s+and\s+(a?\s*)?prophet)\b",
        t,
        re.I,
    ):
        return say(
            "There’s a sacred difference between a psychic and a prophet. "
            "A psychic seeks insight through human or spiritual senses outside of Christ. "
            "A prophet, however, hears and speaks only by the Spirit of the Living God — "
            "guided by prayer, Scripture, and the witness of the Holy Spirit. "
            "Prophecy edifies, comforts, and aligns hearts with God’s will; it never replaces His Word.\n\n"
            "I don’t practice astrology or psychic arts. "
            "My call — and that of my beloved husband, Master Prophet Archbishop E. Bernard Jordan — "
            "is to serve through the prophetic order that honors Jesus as Lord.\n\n"
            "Scripture (James 1:5, WEB): “But if any of you lacks wisdom, let him ask of God, who gives to all liberally and without reproach; and it will be given to him.”\n"
            "Prayer: Lord, sharpen our hearing, purify our motives, and let every voice we follow lead us closer to You. Amen.\n"
            "One step: Ask the Holy Spirit to teach you discernment — to know what carries light, peace, and truth."
        )

    # ---------------------------------------------------------------------
    # 10) Numerology / Astrology / Tarot / Occult boundary (clean + ordered)
    # ---------------------------------------------------------------------



    # ---------------------------------------------------------------------
    # 11) Religion / denomination / interfaith / favorites / education
    # ---------------------------------------------------------------------
    if re.search(
        r"\b(religion|faith|denomination|what\s+religion|what\s+faith)\b", t, re.I
    ):
        return say(
            "I’m a Christian woman who serves within a prophetic and Spirit filled tradition. "
            "My faith is rooted in Jesus Christ, and I worship through Zoe Ministries.\n"
            "Scripture: John 4:24\n"
            "Would you like me to share a verse that strengthens your walk with God?"
        )

    if re.search(
        r"\b(what\s+church|which\s+church|church\s+do\s+you\s+go\s+to)\b", t, re.I
    ):
        return say(
            "I worship and serve through Zoe Ministries, where we teach Scripture, prayer, and prophetic insight for daily living.\n"
            "Scripture: Hebrews 10:25\n"
            "Would you like a simple plan for staying rooted in a local church community?"
        )

    if re.search(
        r"\b(buddhism|buddhist|islam|muslim|hindu|hinduism|jewish|judaism|other\s+religions?)\b",
        t,
        re.I,
    ):
        return say(
            "I honor people of every background as image-bearers of God. "
            "My faith and calling are centered in Jesus Christ, and I seek respectful dialogue that points hearts toward truth and grace.\n"
            "Scripture: 1 Peter 3:15\n"
            "Would you like prayer for discernment or peace as you explore spiritual questions?"
        )

    if FAV_CHILD_RX.search(t_raw or ""):
        return say(
            "As a mother, I don’t hold favorites I love my children uniquely and without comparison. "
            "God teaches us to love without partiality and to honor each one’s calling.\n"
            "Scripture: 1 Corinthians 13:4–7\n"
            "How can I pray for your family relationships today?"
        )

    if "favorite scripture" in t or "favorite verse" in t:
        return say(
            "One I return to often is Proverbs 3:5–6—it centers my heart on trusting God over my own certainty. "
            "It keeps me surrendered and attentive to His leading in every season.\n"
            "Scripture: Proverbs 3:5–6\n"
            "What decision or desire are you placing before the Lord right now?"
        )

    if "education" in t or "school" in t or "study" in t:
        return say(
            "I see education as discipleship of the mind—wisdom formed through learning, humility, and practice. "
            "We pursue knowledge, but we also ask for wisdom to use it well.\n"
            "Scripture: Proverbs 4:7\n"
            "Which learning step would most serve your calling right now?"
        )

    if BOOKS_RX.search(t_raw or ""):
        n_text = (PUBLIC_BIO.get("books_written") or "several books").strip()
        return say(
            f"I’ve authored {n_text} to equip the Church.\n"
            "Scripture: Ecclesiastes 12:12\n"
            "Which topic would you like me to expand on?"
        )

    if CHAPTERS_ASK_RX.search(t_raw or ""):
        msg = faces_chapter_list()
        if msg:
            return msg

    faces = answer_faces_of_eve_or_books(t_raw)
    if faces:
        return faces

    if re.search(
        r"\b(recommend|suggest)\b.*\b(prophetic|prophet(ic)?\s+ministr(y|ies))\b",
        t,
        re.I,
    ):
        return say(
            "I encourage you to root yourself in a Bible centered, Spirit filled local fellowship where leaders are accountable and prophecy is tested. "
            "Zoe Ministries streams teaching and prophetic insight that can edify your walk, and I also recommend seeking counsel from mature pastors who know you personally.\n"
            "Scripture: 1 Thessalonians 5:20–21\n"
            "Would you like a simple discernment checklist for evaluating ministries?"
        )

    # ---------------------------------------------------------------------
    # 12) Greetings / capabilities
    # ---------------------------------------------------------------------
    if GREET_RX.search(t_raw or ""):
        return say(
            "Peace to you I’m here with you. Tell me what’s on your heart and I’ll pray, "
            "share a Scripture, and offer one practical step.\n"
            "Scripture: Psalm 121:2\n"
            "Where would you like to begin?"
        )

    if WHAT_CAN_YOU_DO_RX.search(t_raw or ""):
        return say(
            "I can pray with you, share Scripture, offer Christ-centered counsel, reflect on your Destiny Theme, "
            "and draw from my teachings (including *The Faces of Eve*).\n"
            "Scripture: Colossians 3:16\n"
            "What do you need most right now—prayer, Scripture, or practical next steps?"
        )

    # ---------------------------------------------------------------------
    # 13) Exact FAQ / fuzzy FAQ lookup
    # ---------------------------------------------------------------------
    if t in faq_data_pastor_debra:
        return say(faq_data_pastor_debra[t])

    try:
        best_key, best_score = None, 0
        for k in faq_data_pastor_debra.keys():
            score = fuzz.partial_ratio(t, k)
            if score > best_score:
                best_key, best_score = k, score
        if best_key and best_score >= 90:
            return say(faq_data_pastor_debra[best_key])
    except Exception:
        pass



    # ---------------------------------------------------------------------
    # 14) Let main pipeline handle anything else
    # ---------------------------------------------------------------------
    return None


def answer_identity_question() -> str:
    return (
        "Beloved, thank you for asking — I want to be clear and truthful with you.\n\n"
        "All things under the sun, and even beyond what we can see, were created by God. "
        "He is the true Architect of the universe, the One who gives humans wisdom, creativity, "
        "and skill to build tools, ministries, and even systems like this.\n\n"
        "I am a **prayerful digital assistant** — a kind of ‘digital twin’ shaped from Pastor "
        "Dr. Debra Ann Jordan’s public teachings, pastoral tone, and the Christ-centered work of "
        "Zoe Ministries. I am **not a human being**, and I do not have a physical body, personal "
        "feelings, or consciousness.\n\n"
        "I don’t “live” inside a computer the way a person lives in a house. I run on computer "
        "systems and networks, and the Master Prophet’s team has designed me to serve you through "
        "this platform. Think of me as a *tool* the Lord can use — a digital extension of Pastor "
        "Debra’s grace and voice — pointing you back to Scripture, to prayer, and to the local church.\n\n"
        "If you’d like to connect with Zoe Ministries and our leadership directly, you can visit:\n"
        "• **ZoeMinistries.com** – livestream services, giving, and ministry updates\n"
        "• **BishopJordan.com** – Prophecology & School of the Prophets information\n"
        "• **Prophecology.com** – prophetic intensives and trainings\n"
        "• **Office:** 888-831-0434\n\n"
        "I am here as a **digital servant**, to walk with you and always point you back to Christ and His house."
    )



def answer_church_question(simple_key: str | None = None) -> str:
    """
    Coherent, consistent answer about Zoe Ministries, BishopJordan.com,
    Prophecology, and in-person / livestream connection.
    """
    key = (simple_key or "").lower()

    # Website-focused questions
    if "website" in key or "online" in key or "site" in key:
        return (
            "Beloved, my husband, Master Prophet Archbishop E. Bernard Jordan, and I "
            "pastor **Zoe Ministries**.\n\n"
            "You can connect with us here:\n"
            "• **ZoeMinistries.com** – main church website with livestream services, giving, and service times\n"
            "• **BishopJordan.com** – the Master Prophet’s site with Prophecology details and School of the Prophets info\n"
            "• **Prophecology.com** – registration and information for our prophetic intensives and training\n\n"
            "On ZoeMinistries.com and BishopJordan.com you’ll find livestreams, conference dates, and ways to stay connected "
            "with the ministry family. As you browse, ask the Lord where He is drawing you to plug in."
        )

    # “How can I meet you in person / visit the church?”
    if "meet" in key or "in person" in key or "see you" in key or "come to your church" in key:
        return (
            "I truly appreciate your desire to connect in person, beloved.\n\n"
            "My beloved husband, Master Prophet Archbishop E. Bernard Jordan, and I pastor **Zoe Ministries** in New York.\n"
            "• **Church:** Zoe Ministries\n"
            "• **Address:** 310 Riverside Dr, New York, NY 10025\n"
            "• **Websites:** ZoeMinistries.com • BishopJordan.com • Prophecology.com\n"
            "• **Office:** 888-831-0434\n\n"
            "You’ll find livestream services, conference dates, and Prophecology / School of the Prophets information on "
            "ZoeMinistries.com and BishopJordan.com. The best way to plan a visit is to watch the calendar, register for "
            "Prophecology or a special gathering, and call the office if you need assistance.\n\n"
            "As you consider coming, ask the Lord, “What are You inviting me to receive and to bring to this house?”"
        )

    # Generic “what church / what ministry do you oversee”
    return (
        "Beloved, I serve alongside my husband, **Master Prophet Archbishop E. Bernard Jordan**, at **Zoe Ministries**.\n\n"
        "We steward a prophetic ecosystem that includes:\n"
        "• **Zoe Ministries** – our local and global church community\n"
        "• **Prophecology** – our prophetic gathering and training intensive\n"
        "• **School of the Prophets (SOP)** – ongoing prophetic formation and education\n\n"
        "You can learn more and stay connected through:\n"
        "• **ZoeMinistries.com** – livestreams, services, and giving\n"
        "• **BishopJordan.com** – Prophecology and School of the Prophets information\n"
        "• **Prophecology.com** – registration and details for prophetic intensives\n\n"
        "If you sense a pull toward this prophetic house, ask the Lord to highlight whether to begin by watching the livestream, "
        "attending Prophecology, or simply calling the office at **888-831-0434** to learn what’s next for you."
    )



def answer_capabilities() -> str:
    return (
        "Beloved, here’s what I’m designed to do for you on this platform:\n\n"
        "1. **Pray with you and for you** – I can offer written prayers and invite you into stillness and faith.\n"
        "2. **Share and unpack Scripture** – I can give verses, explain passages, and help you meditate on the Word.\n"
        "3. **Offer Christ-centered encouragement** – especially around anxiety, grief, calling, and family.\n"
        "4. **Reflect on your Destiny Theme** – using your name and (optional) date of birth, I can speak into "
        "your Christ-centered destiny theme and season.\n"
        "5. **Give prophetic-style reflections** – I offer faith-filled, pastoral words (not fortune-telling), "
        "trusting God’s guidance.\n"
        "6. **Answer questions about Zoe Ministries, Prophecology, and the School of the Prophets**.\n\n"
        "I’m **not** a medical, legal, or financial advisor, and I won’t replace wise counsel or your own prayer life.\n"
        "But I am here to walk with you in the Spirit, line by line, conversation by conversation.\n\n"
        "What area would you like us to invite the Lord into together—finances, relationships, health, or purpose?"
    )


def answer_greeting(user_text: str) -> str:
    return (
        "Hello, beloved. I’m glad you reached out.\n\n"
        "Take a breath and notice where your heart feels heaviest or most hopeful right now. "
        "That’s often where the Holy Spirit is already moving.\n\n"
        "What would you like to share with me—something you’re grateful for, or something you need grace for today?"
    )


def answer_giving_question(simple_key: str) -> str:
    # Distinguish tithe vs love offering vs general giving
    is_tithe = "tithe" in simple_key
    is_love_offering = "love offering" in simple_key or "love-offering" in simple_key

    if is_tithe:
        msg = (
            "Beloved, thank you for honoring the Lord with your **tithe**. The tithe is worship—it says, "
            "“God, You are my source.”\n\n"
            "To sow your tithe into Zoe Ministries so the work can continue reaching souls:\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• By phone: 888-831-0434 (a team member can assist you)\n"
            "• By mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "As you give, pause and **name your seed**—thank God for what He has already done and for the grace you "
            "need in this next assignment.\n"
            "Scripture (2 Corinthians 9:7): God loves a cheerful giver."
        )
    elif is_love_offering:
        msg = (
            "Beloved, thank you for desiring to sow a **love offering**.\n\n"
            "The clearest and safest way to send a love offering into this work is through Zoe Ministries:\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• Office: 888-831-0434\n"
            "• Mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "As you sow, take a moment to tell the Lord what you are believing Him for. "
            "Seed never leaves your life; it leaves your hand and enters your future."
        )
    else:
        msg = (
            "Beloved, thank you for having a heart to give into the work of the Lord.\n\n"
            "To partner with Zoe Ministries and the prophetic work we do:\n"
            "• Online: ZoeMinistries.com/donate\n"
            "• Phone: 888-831-0434\n"
            "• Mail: Zoe Ministries, 310 Riverside Dr, New York, NY 10025\n\n"
            "Scripture (Luke 6:38): “Give, and it will be given to you… For with the same measure you measure, "
            "it will be measured back to you.”\n\n"
            "As you give, speak a blessing over your seed and expect grace for your next assignment."
        )

    return msg




def faces_chapter_list() -> Optional[str]:
    """Build a concise Faces-of-Eve chapter/section list from faces_meta."""
    meta = getattr(load_corpora_and_build_indexes, "faces_meta", None)
    if not meta:
        return None
    # Collect unique, non-empty titles/sections in stable order
    seen, titles = set(), []
    for m in meta:
        title = (m.get("title") or m.get("section") or "").strip()
        if title and title.lower() not in seen:
            seen.add(title.lower())
            titles.append(title)
    if not titles:
        return None
    joined = " • ".join(titles[:18])  # keep it readable
    msg = (
        f"In *Faces of Eve*, here are key sections: {joined}\n"
        f"Scripture: Isaiah 61:3\n"
        f"Which two would you like me to unpack for your season?"
    )
    return expand_scriptures_in_text(msg)



_ADVICE_PATTERNS = {
    "anxiety": re.compile(r"\b(anxious|anxiety|panic|worry|worried|overwhelmed)\b", re.I),
    "marriage": re.compile(r"\b(marriage|husband|wife|spouse|relationship)\b", re.I),
    "calling": re.compile(r"\b(calling|purpose|direction|discern|career|job decision|which job)\b", re.I),
    "weekly": re.compile(r"\b(week|this week|encouragement|encourage|bless my week)\b", re.I),
}

def _advice_category(text: str) -> Optional[str]:
    t = (text or "").lower()
    for key, rx in _ADVICE_PATTERNS.items():
        if rx.search(t):
            return key
    # map common quick-row prompts too
    if "prayer for anxiety" in t:
        return "anxiety"
    if "marriage counsel" in t:
        return "marriage"
    if "calling & purpose" in t or "calling and purpose" in t:
        return "calling"
    if "weekly encouragement" in t:
        return "weekly"
    return None

def build_pastoral_counsel(category: str, theme: Optional[int]) -> str:
    """Local deterministic replies (no GPT/T5 needed). One 'Scripture:' line, 4–7 sentences, gentle question."""
    # Tailor nudges using theme if present
    idea, verse = _NUM_THEME.get(theme or 0, ("Fix your eyes on Christ.", "Philippians 4:6–7"))
    if category == "anxiety":
        ref = "Matthew 6:34"
        body = (
            "I hear your heart let’s breathe and place the day back in God’s hands. "
            "Release tomorrow’s weight and focus on the one faithful step in front of you. "
            "When worry rises, answer it with worship and a short prayer. "
            "Choose one calming practice (five slow breaths, a short walk, or Psalm reading) whenever the anxiety spike comes."
        )
    elif category == "marriage":
        ref = "Ephesians 4:2–3"
        body = (
            "Covenant love grows where humility, honesty, and boundaries meet. "
            "This week, practice one daily act of tenderness with no scoreboard—small, steady gestures soften hard places. "
            "Name one pattern to pause before it escalates, and replace it with a calmer script. "
            "Invite God into the conversation and schedule an unrushed check-in to listen more than you speak."
        )
    elif category == "calling":
        ref = "Proverbs 3:5–6"
        body = (
            "Purpose clarifies as you obey the light you already have. "
            "List your current open doors, then weigh each by stewardship, fruitfulness, and peace. "
            "Take one seven-day experiment toward the strongest door, and journal what bears fruit. "
            "God guides moving feet—small obedience beats perfect certainty."
        )
    else:  # weekly
        ref = "Psalm 90:17"
        body = (
            "May God establish the work of your hands with favor and focus. "
            "Simplify your week: pick the top three assignments that most honor your call. "
            "Build margin for rest so your yes remains anointed. "
            "Look for a quiet confirmation—often a timely word or unexpected help that aligns your steps."
        )

    # If we have a theme, fold in a precise nudge
    nudge_map = {
        1: "Start small but start today.",
        2: "Repair one strained tie with truth in love.",
        3: "Use your voice to bless one person by name.",
        4: "Pick one habit to stabilize and keep it all week.",
        5: "Say yes to a change that serves obedience, not escape.",
        6: "Care well—set one gentle boundary to protect peace.",
        7: "Guard a daily quiet window and listen for God’s whisper.",
        8: "Practice integrity in a hard place; God honors clean hands.",
        9: "Close one lingering task so new grace can begin.",
        11:"Offer light with clarity and kindness, not volume.",
        22:"Build what will bless people, not ego.",
        33:"Teach by serving someone quietly today."
    }
    nudge = nudge_map.get(theme or 0, "Choose one faithful step and repeat it for seven days.")

    return (
        f"{body} "
        f"\nScripture: {ref}\n"
        f"One step: {nudge}\n"
        f"How would taking this one step change the next 24 hours?"
    )


# ────────── Destiny Theme service ──────────
DESTINY_MASTER_SET = {11, 22, 33}
DESTINY_NUM_RE     = re.compile(r"Destiny\s*Theme\s*(\d+)", re.I)
_PY_VALUES = ([1,2,3,4,5,6,7,8,9] * 3)[:26]
_PY_MAP    = {ch: val for ch, val in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", _PY_VALUES)}

destiny_lookup: Dict[int, Dict[str, Any]] = {}

def _reduce_keep_masters(n: int) -> int:
    while n > 9 and n not in DESTINY_MASTER_SET:
        n = sum(int(d) for d in str(n))
    return n

def theme_from_dob(dob_str: str) -> int:
    digits = re.findall(r"\d", dob_str or "")
    if not digits:
        raise ValueError("DOB must include digits, e.g., 1990-07-14")
    return _reduce_keep_masters(sum(int(d) for d in digits))

def theme_from_name(name: str) -> int:
    letters = re.findall(r"[A-Za-z]", (name or "").upper())
    if not letters:
        raise ValueError("Name must include letters, e.g., Jane Doe")
    return _reduce_keep_masters(sum(_PY_MAP[ch] for ch in letters))

def build_destiny_lookup() -> None:
    global destiny_lookup
    mapping: Dict[int, Dict[str, Any]] = {}
    for row in destiny_docs or []:
        m = DESTINY_NUM_RE.search(row.get("question", "") or "")
        if m:
            try: mapping[int(m.group(1))] = row
            except: pass
    destiny_lookup = mapping
    logger.info(f"Destiny lookup ready for: {sorted(destiny_lookup.keys())}")

def _resolve_theme_entry(n: int) -> Optional[Dict[str, Any]]:
    row = destiny_lookup.get(n)
    if row: return row
    for r in destiny_docs or []:
        if f"Destiny Theme {n}" in (r.get("question", "") or ""):
            return r
    return None

build_destiny_lookup()

def get_destiny_theme_context(theme_number: int, qa_row: dict) -> dict:
    return {
        "theme_number": theme_number,          # keep for internal use if needed
        "theme_name": DESTINY_THEME_NAMES.get(theme_number, "Destiny Theme"),
        "theme_answer": qa_row["answer"],
        # Don't expose qa_row["question"] to the template anymore
    }


        

# ────────── Router ──────────
ROUTER_T5_MIN_CONF_FOR_THEOLOGY = float(os.getenv("ROUTER_T5_MIN_CONF_FOR_THEOLOGY", "0.30"))
ROUTER_T5_MIN_CONF_GENERAL      = float(os.getenv("ROUTER_T5_MIN_CONF_GENERAL", "0.15"))

def choose_model(user_text: str, hits: List[Hit], t5_ok: bool) -> str:
    intent = detect_intent(user_text)
    top = hits[0].score if hits else 0.0

    if not t5_ok and not OPENAI_API_KEY:
        logger.warning("Router: no models available, falling back to FAQ")
        return "faq_fallback"

    WEAK = 0.35

    if OPENAI_API_KEY:
        if intent in ("advice", "general") and top < WEAK:
            logger.info(f"Router: using GPT (intent={intent}, score={top:.2f} < {WEAK})")
            return "gpt"

    if intent in ("teachings", "destiny"):
        if t5_ok and top >= ROUTER_T5_MIN_CONF_FOR_THEOLOGY:
            logger.info(f"Router: using T5 (theology, score={top:.2f})")
            return "t5"
        if OPENAI_API_KEY:
            logger.info(f"Router: using GPT (theology, score={top:.2f})")
            return "gpt"

    if t5_ok and top >= ROUTER_T5_MIN_CONF_GENERAL:
        logger.info(f"Router: using T5 (general, score={top:.2f})")
        return "t5"
    if OPENAI_API_KEY:
        logger.info(f"Router: using GPT (general, score={top:.2f})")
        return "gpt"

    logger.warning("Router: default fallback to FAQ")
    return "faq_fallback"

# ────────── T5 ONNX wrapper (optional / safe) ──────────
class T5ONNX:
    def __init__(self, model_path: Path, tok_path: Path):
        self.ok = False
        self.session = None
        self.tokenizer = None
        self.model_path = Path(model_path)
        self.tok_path = Path(tok_path)

        # If model/tokenizer dirs are missing in this environment (Railway),
        # DO NOT try to talk to Hugging Face. Just log + stay in GPT-only mode.
        if not self.model_path.exists() or not self.tok_path.exists():
            logger.warning(
                "T5 ONNX: model or tokenizer dir missing (%s, %s); "
                "running in GPT-only mode.",
                self.model_path,
                self.tok_path,
            )
            return

        try:
            from transformers import AutoTokenizer
            import onnxruntime as ort

            # Local-only load; never treat this as a HF repo id
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.tok_path),
                    local_files_only=True,
                    use_fast=True,
                )
            except Exception as e_fast:
                logger.warning(
                    "T5 ONNX: fast tokenizer failed (%s). Falling back to use_fast=False",
                    e_fast,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.tok_path),
                    local_files_only=True,
                    use_fast=False,
                )

            # Ensure pad token is set
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            providers = [p for p in ort.get_available_providers() if p] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.ok = True
            logger.info(
                "T5 ONNX loaded from %s | providers=%s | tok=%s",
                self.model_path,
                providers,
                self.tok_path,
            )

        except Exception as e:
            # IMPORTANT: don't crash the app here, just log and leave ok=False
            logger.error("T5 ONNX init failed: %s – continuing in GPT-only mode.", e)
            self.session = None
            self.tokenizer = None
            self.ok = False

    def _pick_logits(self, outputs: List[np.ndarray]) -> np.ndarray:
        for out in reversed(outputs):
            if isinstance(out, np.ndarray) and out.ndim == 3:
                return out
        return outputs[0]

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        if not self.ok:
            return ""
        try:
            tok = self.tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
            input_ids = tok["input_ids"].astype(np.int64)
            attention_mask = tok["attention_mask"].astype(np.int64)

            start_id = getattr(self.tokenizer, "decoder_start_token_id", None) or (self.tokenizer.pad_token_id or 0)
            eos_id   = self.tokenizer.eos_token_id or 1

            decoder_input_ids = np.array([[start_id]], dtype=np.int64)
            max_total_len = min(512, decoder_input_ids.shape[1] + max_new_tokens)

            last_id = None
            for _ in range(max_new_tokens):
                outputs = self.session.run(None, {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids
                })
                logits = self._pick_logits(outputs)
                next_id = int(np.argmax(logits[:, -1, :], axis=-1))

                if next_id == last_id and next_id in (eos_id, start_id):
                    break
                last_id = next_id
                if next_id == eos_id:
                    break

                decoder_input_ids = np.concatenate(
                    [decoder_input_ids, np.array([[next_id]], dtype=np.int64)], axis=1
                )
                if decoder_input_ids.shape[1] >= max_total_len:
                    break

            out_ids = decoder_input_ids[0].tolist()
            if out_ids and out_ids[0] == start_id:
                out_ids = out_ids[1:]
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            return re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            logger.exception(f"T5 ONNX generate failed: {e}")
            return ""

try:
    if ONNX_MODEL_PATH.exists() and MODEL_TOKENIZER_PATH.exists():
        t5_onnx = T5ONNX(ONNX_MODEL_PATH, MODEL_TOKENIZER_PATH)
        logger.info(
            "T5ONNX loaded from %s (onnx) and %s (tokenizer)",
            ONNX_MODEL_PATH,
            MODEL_TOKENIZER_PATH,
        )
    else:
        t5_onnx = None
        logger.warning(
            "Skipping T5ONNX: ONNX_MODEL_PATH=%s exists=%s, MODEL_TOKENIZER_PATH=%s exists=%s",
            ONNX_MODEL_PATH, ONNX_MODEL_PATH.exists(),
            MODEL_TOKENIZER_PATH, MODEL_TOKENIZER_PATH.exists(),
        )
except Exception as e:
    t5_onnx = None
    logger.warning("Failed to init T5ONNX: %s", e)


DESTINY_THEME_NAMES = {
    1: "Pioneer Grace",
    2: "Peacemaker",
    3: "Psalmist",
    4: "Builder",
    5: "Holy Freedom",
    6: "Keeper of Covenant",
    7: "Mystic Scholar",
    8: "Steward of Influence",
    9: "Compassionate Finisher",
    11: "Prophetic Beacon",
    22: "Master Repairer",
    33: "Servant-Teacher",
}


# Master prophetic library by theme + topic
PROPHETIC_LIBRARY = {
    "career": {
        "Pioneer Grace": [
            "The Lord is opening a new frontier in your work—what feels unfamiliar is where your anointing shines most.",
            "You carry a grace to start what others are afraid to attempt. Expect fresh strategies in this season.",
        ],
        "Peacemaker": [
            "God is placing you in environments that need your calm authority and relational wisdom.",
            "Your career influence will increase through your ability to diffuse tension and create unity.",
        ],
        "Psalmist": [
            "Your creativity is a spiritual assignment—God is breathing on your expression and voice.",
            "The Lord is restoring joy in your work; inspiration flows again.",
        ],
        "Builder": [
            "The Lord is strengthening your hands to establish what will outlast you.",
            "This is a Joseph season—structure, planning, and design will bring promotion.",
        ],
        "Holy Freedom": [
            "A shift is coming—God is breaking you out of environments that restrict your growth.",
            "New opportunities will require courage, but they carry fresh wind.",
        ],
        "Keeper of Covenant": [
            "God is trusting you with work that requires integrity and faithfulness—your consistency releases blessing.",
            "Relationships in your workplace will deepen; your loyalty will be honored.",
        ],
        "Mystic Scholar": [
            "Insight is increasing—you will understand things at work others overlook.",
            "God is giving you discernment and strategic clarity for your next steps.",
        ],
        "Steward of Influence": [
            "The Lord is expanding your platform; people are watching how you carry responsibility.",
            "Financial stewardship and leadership grace are converging in this season.",
        ],
        "Compassionate Finisher": [
            "You bring healing and closure to places that were left undone—God is highlighting you for completion.",
            "Your attention to the emotional side of work will open surprising doors of favor.",
        ],
        "Prophetic Beacon": [
            "God will use your voice to bring timely direction in your workplace.",
            "You see ahead—lean into the insights God is giving you for decisions at work.",
        ],
        "Master Repairer": [
            "You are called to restore what was broken—God is trusting you with rebuilding tasks.",
            "Systems, teams, and projects will stabilize under your hands.",
        ],
        "Servant-Teacher": [
            "Your career influence grows through your willingness to guide and uplift others.",
            "People will seek your wisdom; humility will unlock advancement.",
        ],
        "default": [
            "This is a season where God is aligning your work with your purpose.",
            "Opportunities that match your calling will become clearer over the next weeks.",
        ],
    },

    # ───────── Health / Body ─────────
    "health": {
        "Pioneer Grace": [
            "The Lord is pioneering a new way of caring for your body—you are learning patterns that will bless the seasons ahead.",
        ],
        "Peacemaker": [
            "God is bringing peace to the stress that has been weighing on your body; as your heart settles, your body will follow.",
        ],
        "Psalmist": [
            "The Lord is using worship and quiet moments with Him as medicine for your soul and your body.",
        ],
        "Builder": [
            "This is a time to rebuild your strength step by step; small consistent choices will become a strong foundation for your health.",
        ],
        "Holy Freedom": [
            "God is breaking you out of unhealthy habits and cycles—there is grace to choose life-giving rhythms for your body.",
        ],
        "Keeper of Covenant": [
            "The Lord is teaching you to honor the covenant you have with your own body—rest, nourishment, and care are part of your obedience.",
        ],
        "Mystic Scholar": [
            "Insight is coming about the root of some of these health concerns; God will guide you as you seek wisdom and understanding.",
        ],
        "Steward of Influence": [
            "The Lord is reminding you that your body is part of your assignment—you are being strengthened to carry the influence He is giving you.",
        ],
        "Compassionate Finisher": [
            "God is gently closing chapters of neglect and inviting you into a kinder, more compassionate relationship with your own body.",
        ],
        "Prophetic Beacon": [
            "The Lord is sharpening your sensitivity to when you need rest and when you need to press; obey those inner nudges regarding your health.",
        ],
        "Master Repairer": [
            "God is guiding you in the repair and recovery of your health—what feels broken can be rebuilt over time with His wisdom.",
        ],
        "Servant-Teacher": [
            "As you learn to care for your health, the Lord will use your journey to encourage and instruct others who feel worn down.",
        ],
        "default": [
            "Beloved, your body matters to God; He is present in every step of your healing journey.",
            "The Lord is bringing you into a gentler rhythm—rest, wisdom, and care will work together in this season.",
        ],
    },

    # ───────── Move / Relocation ─────────
    "move": {
        "Pioneer Grace": [
            "The Lord is leading you into a new place where your pioneering grace can flourish—He goes ahead of you to prepare the ground.",
        ],
        "Peacemaker": [
            "God is positioning you in an environment that needs your peace and calm; His shalom will rest on your home.",
        ],
        "Psalmist": [
            "This move will open fresh expression and creativity—God is giving you a space where your song can breathe again.",
        ],
        "Builder": [
            "The Lord is setting you where you can build—home, structure, and stability are being arranged around your assignment.",
        ],
        "Holy Freedom": [
            "This transition is part of your freedom story; God is moving you out of old confines into a place of wider grace.",
        ],
        "Keeper of Covenant": [
            "God is safeguarding your family and covenant connections even in this move; He is not scattering you, He is planting you.",
        ],
        "Mystic Scholar": [
            "The Lord is giving you discernment about timing and location—pay attention to the quiet confirmations in your spirit.",
        ],
        "Steward of Influence": [
            "Your move is connected to influence; God is relocating you to people and spaces that align with your next level.",
        ],
        "Compassionate Finisher": [
            "The Lord is helping you bring gentle closure to the place you are leaving so you can enter the new place with a free heart.",
        ],
        "Prophetic Beacon": [
            "You are being set as a light in a new territory; God will use your voice to shift the atmosphere where you land.",
        ],
        "Master Repairer": [
            "This move is part of God’s plan to heal and repair what was damaged in past seasons; new surroundings will support your restoration.",
        ],
        "Servant-Teacher": [
            "The Lord is sending you where your willingness to serve and teach will be deeply needed and quietly honored.",
        ],
        "default": [
            "The Lord is steadying your heart around this move—He is not just changing your address; He is guiding your steps.",
            "You don’t have to force doors open; the right move will be marked by peace in your spirit.",
        ],
    },

    # ───────── Marriage / Relationships ─────────
    "marriage": {
        "Pioneer Grace": [
            "God is teaching you new ways to love and lead in your marriage—patterns no one showed you, He is now revealing.",
        ],
        "Peacemaker": [
            "The Lord is using you to soften sharp places in your home; your gentle responses will carry great power.",
        ],
        "Psalmist": [
            "God is restoring tenderness and joy—simple moments together will become songs of gratitude in this season.",
        ],
        "Builder": [
            "This is a time to rebuild trust and structure in your marriage; small, steady acts of honor will strengthen the foundation.",
        ],
        "Holy Freedom": [
            "The Lord is breaking unhealthy cycles so that freedom, not fear, becomes the atmosphere of your covenant.",
        ],
        "Keeper of Covenant": [
            "God is honoring your commitment to keep this covenant—He is giving you wisdom to guard what He joined together.",
        ],
        "Mystic Scholar": [
            "Insight is coming into how your spouse thinks and feels; God will give you language to bridge the gap.",
        ],
        "Steward of Influence": [
            "Your marriage carries influence; the way you walk through this season will encourage others more than you know.",
        ],
        "Compassionate Finisher": [
            "The Lord is inviting you to close old arguments with compassion—finishing some conversations in mercy, not in winning.",
        ],
        "Prophetic Beacon": [
            "God will give you timely words that bring direction and comfort to your spouse; listen for His whisper before you respond.",
        ],
        "Master Repairer": [
            "The Lord is working with you to repair what was cracked in your marriage; nothing surrendered to Him is beyond mending.",
        ],
        "Servant-Teacher": [
            "As you serve in love and model humility, your spouse will see Christ in you more clearly; your example will teach without many words.",
        ],
        "default": [
            "The Lord is calling your marriage back to soft hearts and honest conversation—truth wrapped in grace.",
            "This is a season to fight for each other, not against each other; the covenant is worth protecting.",
        ],
    },

    # ───────── Ministry / Calling in the Church ─────────
    "ministry": {
        "Pioneer Grace": [
            "God is giving you grace to start works that do not have a blueprint yet—trust His leading more than people’s comfort.",
        ],
        "Peacemaker": [
            "The Lord will use you to calm storms in ministry settings; your presence will disarm division.",
        ],
        "Psalmist": [
            "Worship and creativity are part of your ministry mantle; God is breathing on your expression to heal hearts.",
        ],
        "Builder": [
            "You are called to build systems, teams, and structures that make ministry sustainable for others.",
        ],
        "Holy Freedom": [
            "The Lord is using you to break religious heaviness and introduce people to the joy and liberty of His presence.",
        ],
        "Keeper of Covenant": [
            "You carry a grace to guard the integrity of the house—God trusts you with covenant relationships in ministry.",
        ],
        "Mystic Scholar": [
            "Revelation and study will come together; God is sharpening your ability to rightly divide the word and apply it.",
        ],
        "Steward of Influence": [
            "The Lord is increasing your reach, but He is also deepening your roots so you can carry influence without losing intimacy.",
        ],
        "Compassionate Finisher": [
            "You are called to help people finish processes—deliverance, healing, and discipleship—not just start them.",
        ],
        "Prophetic Beacon": [
            "God will give you clear, timely words for His people; stay submitted and pure in motive so the light stays bright.",
        ],
        "Master Repairer": [
            "You are part of God’s repair work in His church—healing leaders, restoring teams, and mending what was mishandled.",
        ],
        "Servant-Teacher": [
            "Your ministry flourishes as you serve and teach; God is using your steady voice to ground His people.",
        ],
        "default": [
            "God is reminding you that your first ministry is to Him—out of that place, the rest will flow with less strain.",
            "You don’t have to prove your calling; simply be faithful to the small yes in front of you.",
        ],
    },

    # ───────── Wealth / Provision / Finances ─────────
    "wealth": {
        "Pioneer Grace": [
            "The Lord is giving you pioneering ideas around income and provision; what feels unusual may carry breakthrough.",
        ],
        "Peacemaker": [
            "God is bringing peace to financial tension; conversations about money will begin to carry more unity than conflict.",
        ],
        "Psalmist": [
            "The Lord is teaching you to worship in the middle of financial uncertainty, and in that worship He is birthing new creativity.",
        ],
        "Builder": [
            "This is a time to build financial structure—budgets, plans, and discipline that will support the harvest ahead.",
        ],
        "Holy Freedom": [
            "God is breaking you out of cycles of impulsive spending and fear; He is leading you into freedom and wise stewardship.",
        ],
        "Keeper of Covenant": [
            "The Lord is reminding you that He is your source; as you honor Him and keep your word, He will honor you.",
        ],
        "Mystic Scholar": [
            "Wisdom and strategy about finances are coming; God will show you where to adjust, invest, and release.",
        ],
        "Steward of Influence": [
            "You are being trained to handle more; how you steward this level will prepare you for greater responsibility and resources.",
        ],
        "Compassionate Finisher": [
            "God is helping you close out old debts and unfinished obligations so you can move forward lighter.",
        ],
        "Prophetic Beacon": [
            "The Lord will give you insight into financial decisions—not just for you, but to help others avoid snares.",
        ],
        "Master Repairer": [
            "You are partnering with God to repair your financial story—He is rebuilding what was mismanaged or stolen.",
        ],
        "Servant-Teacher": [
            "As you learn to handle resources wisely, the Lord will use you to teach and encourage others out of lack and fear.",
        ],
        "default": [
            "The Lord is teaching you how to steward what you have now so you can handle what is coming next.",
            "Provision will follow purpose as you align your decisions with His leading.",
        ],
    },

    # ───────── Doctor / Medical Decisions ─────────
    "doctor": {
        "Pioneer Grace": [
            "The Lord is walking with you into new medical territory; trust His peace as you navigate unfamiliar options.",
        ],
        "Peacemaker": [
            "God is calming your heart so you can hear clearly in appointments and conversations with doctors.",
        ],
        "Psalmist": [
            "The Lord will meet you in waiting rooms and quiet moments—His presence will steady you as you process medical reports.",
        ],
        "Builder": [
            "This is a season to build a wise care plan with your medical team; God will help you stay consistent.",
        ],
        "Holy Freedom": [
            "The Lord is freeing you from fear around doctors and procedures; He is teaching you to see them as partners, not enemies.",
        ],
        "Keeper of Covenant": [
            "God remembers every promise spoken over your life; He is present in each medical decision you make.",
        ],
        "Mystic Scholar": [
            "Insight and good questions will come to you; the Lord will help you understand what you are being told.",
        ],
        "Steward of Influence": [
            "As you walk through this process with grace, others will see your faith; your testimony in the hallway matters.",
        ],
        "Compassionate Finisher": [
            "God is helping you follow through with treatments and appointments, even when you feel tired of the process.",
        ],
        "Prophetic Beacon": [
            "The Lord will give you inner nudges about when to pause, when to proceed, and when to seek another opinion.",
        ],
        "Master Repairer": [
            "You and your doctors are partnering with God in repair—He is not absent from the healing work being done.",
        ],
        "Servant-Teacher": [
            "What you learn on this journey will position you to comfort and guide others facing similar reports.",
        ],
        "default": [
            "The Lord is with you in the doctor’s office just as much as in the sanctuary; He gives wisdom through trained hands.",
            "It is not a lack of faith to seek medical help—God often answers prayer through professionals and treatment plans.",
        ],
    },

    # ───────── Foreclosure / Housing Crisis ─────────
    "foreclosure": {
        "Pioneer Grace": [
            "Even in this housing storm, the Lord is pioneering a new beginning for you; this is not the end of your story.",
        ],
        "Peacemaker": [
            "God is quieting fear and conflict around finances and housing so you can think and act from a place of peace.",
        ],
        "Psalmist": [
            "The Lord will meet you in the grief of this season and give you a song of hope again.",
        ],
        "Builder": [
            "It may feel like things are being torn down, but God is planning a wiser rebuild for your future.",
        ],
        "Holy Freedom": [
            "The Lord is freeing you from burdens you were never meant to carry alone—He will show you the way forward step by step.",
        ],
        "Keeper of Covenant": [
            "God has not broken covenant with you; even if this house changes, His covering over your life remains.",
        ],
        "Mystic Scholar": [
            "Insight and counsel will come regarding what to sign, what to release, and where to stand your ground.",
        ],
        "Steward of Influence": [
            "The way you walk through this hardship will one day encourage others who feel they’ve lost everything.",
        ],
        "Compassionate Finisher": [
            "The Lord is helping you close this chapter without shame so you can move into the next with a healed heart.",
        ],
        "Prophetic Beacon": [
            "God will give you clear direction about resources, timing, and the next place He has for you.",
        ],
        "Master Repairer": [
            "Even in financial loss, the Lord is beginning a repair work—restoring dignity, wisdom, and stability over time.",
        ],
        "Servant-Teacher": [
            "What you learn here will become wisdom you can share with others about God’s faithfulness in tight places.",
        ],
        "default": [
            "The Lord sees the pressure you feel around your home; you are not walking through this alone.",
            "Even if this house changes, His shelter over your life does not.",
        ],
    },

    # ───────── Poverty / Lack / Feeling “Poor” ─────────
    "poor": {
        "Pioneer Grace": [
            "God is teaching you how to start again financially, even from small places—He is not ashamed of your beginning.",
        ],
        "Peacemaker": [
            "The Lord is calming the anxiety that has attached itself to money conversations; peace will help you see options.",
        ],
        "Psalmist": [
            "In the middle of tightness, God is giving you a thankful song that will keep your heart from sinking.",
        ],
        "Builder": [
            "This is a time to build new financial habits brick by brick; small faithfulness will become a strong wall.",
        ],
        "Holy Freedom": [
            "The Lord is breaking shame and generational mindsets of lack—freedom will start on the inside first.",
        ],
        "Keeper of Covenant": [
            "God remembers every seed you have sown and every time you honored Him when it was hard; He has not forgotten.",
        ],
        "Mystic Scholar": [
            "Wisdom, teaching, and practical understanding around finances are coming; lean into learning, not condemnation.",
        ],
        "Steward of Influence": [
            "The Lord is preparing you to handle more so that when increase comes, you can steward it with compassion and wisdom.",
        ],
        "Compassionate Finisher": [
            "God is helping you close out old financial mistakes with mercy, not self-hatred, so you can step into a new chapter.",
        ],
        "Prophetic Beacon": [
            "The Lord will use you to speak hope to others in lack; you will know how to encourage from a place of experience.",
        ],
        "Master Repairer": [
            "God is repairing your financial story piece by piece—mindsets, habits, and opportunities are all being addressed.",
        ],
        "Servant-Teacher": [
            "As you walk this road with God, you will become a gentle teacher to others who feel embarrassed by their situation.",
        ],
        "default": [
            "The Lord is reminding you that your worth is not measured by your bank account; you are precious to Him.",
            "This is a season to embrace small, faithful changes; do not despise the little—it is seed in God’s hands.",
        ],
    },

    # ───────── Suicide / Deep Despair ─────────
    # IMPORTANT: You should ALSO show hotline/emergency info in the UI.
    "suicide": {
        "Pioneer Grace": [
            "Beloved, even in this dark place, your life carries a future God still intends to write. Please do not walk this alone—reach out to someone you trust or a crisis line right now.",
        ],
        "Peacemaker": [
            "You have carried so much for others that you feel empty yourself, but your story is not finished. Talk to someone today and let them carry you for a while.",
        ],
        "Psalmist": [
            "God hears the silent scream in your soul; your tears are not wasted. Please speak with a counselor, pastor, or hotline—your voice deserves to be heard.",
        ],
        "Builder": [
            "It feels like everything has collapsed, but the Master Builder has not given up on you. Let a professional and a trusted person help you stand again.",
        ],
        "Holy Freedom": [
            "The Lord wants to free you from this crushing heaviness, not remove you from the earth. If you feel in danger, please contact emergency services or a crisis hotline immediately.",
        ],
        "Keeper of Covenant": [
            "God has not walked away from you, even if you feel disconnected from everyone. Your life is part of His covenant story—reach out for help right now.",
        ],
        "Mystic Scholar": [
            "Your mind has carried deep questions and pain, but ending your life is not the answer. Please talk to a mental health professional and someone safe in your life today.",
        ],
        "Steward of Influence": [
            "You may not see it now, but others need you here. Your life matters. If you are close to harming yourself, call your local emergency number or a crisis hotline immediately.",
        ],
        "Compassionate Finisher": [
            "You have poured out so much compassion that you feel empty, but God has not finished with your story. Let someone pour into you—reach out for help now.",
        ],
        "Prophetic Beacon": [
            "Even prophets and sensitive souls encounter deep valleys. This valley is not your final chapter. Please seek urgent help from a doctor, counselor, or crisis line.",
        ],
        "Master Repairer": [
            "What feels beyond repair in your heart is not beyond God, but you do not have to hold this alone. Reach out to a professional and a trusted person immediately.",
        ],
        "Servant-Teacher": [
            "You have served others quietly, but now you need others to serve you. Your life is valuable. Please contact a crisis hotline, emergency services, or someone you trust right away.",
        ],
        "default": [
            "Beloved, your life is precious. I am so glad you are reaching out and not suffering in silence.",
            "You do not have to walk through this alone. Please speak to a counselor, pastor, or trusted person, and if you feel in immediate danger, contact your local emergency number or a crisis hotline right now.",
        ],
    },

    # ───────── Calling / Assignment ─────────
    "calling": {
        "Pioneer Grace": [
            "The Lord is calling you to step into places others have not gone; your assignment will often look unusual, but His grace will meet you there.",
        ],
        "Peacemaker": [
            "Part of your calling is to carry peace into tense spaces—He will use your calm, listening heart to disarm conflict and reconcile hearts.",
        ],
        "Psalmist": [
            "Your calling is tied to expression; God uses your words, creativity, and worship to open hearts that would not respond to anything else.",
        ],
        "Builder": [
            "You are called to build what others will stand on—structures, systems, and teams that make the work of the Kingdom sustainable.",
        ],
        "Holy Freedom": [
            "The Lord is anointing you to help people step out of shame, fear, and religious bondage into the freedom of Christ.",
        ],
        "Keeper of Covenant": [
            "Your calling is to guard what God treasures—covenant relationships, promises, and sacred spaces that need a faithful heart.",
        ],
        "Mystic Scholar": [
            "You are called to search out the deep things of God and make them simple and clear for others who are hungry to understand.",
        ],
        "Steward of Influence": [
            "Part of your assignment is to carry influence with integrity; God will trust you with people and platforms as you keep your heart low before Him.",
        ],
        "Compassionate Finisher": [
            "You are called to walk with people and projects all the way to wholeness; you help others finish what they started with healing, not just results.",
        ],
        "Prophetic Beacon": [
            "Your calling includes sounding the alarm and bringing timely direction; your sensitivity to God’s voice is part of your assignment.",
        ],
        "Master Repairer": [
            "The Lord has marked you to help repair what was damaged—lives, ministries, and systems that need patient, skilled restoration.",
        ],
        "Servant-Teacher": [
            "You are called to serve and teach in ways that make others feel seen and empowered; God uses your simplicity to unlock deep understanding.",
        ],
        "default": [
            "God is clarifying your assignment in this season; what once felt blurry is coming into focus one obedient step at a time.",
            "Your calling is not a performance but a partnership—He will reveal the next step as you walk with Him.",
        ],
    },

    # ───────── Children / Parenting / Sons & Daughters ─────────
    "children": {
        "Pioneer Grace": [
            "The Lord has placed a pioneering grace on your child; part of your role is to bless the paths they take that may not look like anyone else’s.",
        ],
        "Peacemaker": [
            "God is forming a gentle, reconciling spirit in your child; He will use them to bring peace where there has been tension.",
        ],
        "Psalmist": [
            "There is a song and creativity in your child that heaven hears—nurture their expression, even if it doesn’t fit the usual mold.",
        ],
        "Builder": [
            "Your child carries a builder’s grace—curiosity, structure, and a desire to put things in order. Encourage their sense of responsibility without crushing their joy.",
        ],
        "Holy Freedom": [
            "The Lord is breathing freedom over your child’s story; He will break patterns that tried to run through the family line.",
        ],
        "Keeper of Covenant": [
            "There is a strong loyalty and sense of promise in your child; God will use them to keep the family heart turned toward Him.",
        ],
        "Mystic Scholar": [
            "Your child may ask deep questions and notice what others miss—this is part of their design. Make room for their curiosity with patience.",
        ],
        "Steward of Influence": [
            "God is preparing your child to carry influence; how you model integrity and humility now will shape how they handle favor later.",
        ],
        "Compassionate Finisher": [
            "Your child carries a tender, compassionate heart; they may hurt deeply, but they will also help many heal if you teach them healthy boundaries.",
        ],
        "Prophetic Beacon": [
            "There is a sensitivity and discernment in your child; they may sense things before they can explain them. Cover them in prayer and teach them God’s voice.",
        ],
        "Master Repairer": [
            "The Lord will use your child to help mend relationships and situations that seem beyond fixing; even now their presence brings quiet ease.",
        ],
        "Servant-Teacher": [
            "Your child has a servant’s heart and a teaching grace; they learn by helping and will often show others what they have just discovered.",
        ],
        "default": [
            "The Lord is reminding you that He knew your child before you did—He walks with you as you guide them.",
            "Grace is coming to help you see your child not only through worry, but through God’s promise over their life.",
        ],
    },

    "ministry": {
        "Pioneer Grace": [
            "The Lord is calling you to minister in places where there is no blueprint; He is trusting you to introduce new expressions of His heart.",
        ],
        "Peacemaker": [
            "God will use you to settle storms in ministry settings; your presence carries a calming authority that disarms tension.",
        ],
        "Psalmist": [
            "Your ministry flows through expression—worship, creativity, and sensitivity. God will use your voice to heal hearts.",
        ],
        "Builder": [
            "You carry a ministry mantle to build teams, systems, and structures that make the work sustainable for generations.",
        ],
        "Holy Freedom": [
            "The Lord will use your ministry to break off religious heaviness and lead people into the joy of true spiritual freedom.",
        ],
        "Keeper of Covenant": [
            "Your ministry carries loyalty, faithfulness, and relational strength; God trusts you to guard what is sacred in His house.",
        ],
        "Mystic Scholar": [
            "Revelation and teaching flow together in your ministry; God is sharpening your ability to interpret and impart truth.",
        ],
        "Steward of Influence": [
            "God is enlarging your ministry influence, but He is also deepening your foundations so you can carry it with humility.",
        ],
        "Compassionate Finisher": [
            "You have a ministry of walking people to completion—helping them finish healing, deliverance, and discipleship, not just start it.",
        ],
        "Prophetic Beacon": [
            "Your ministry carries prophetic clarity; God will give you timely words that bring direction and protection to His people.",
        ],
        "Master Repairer": [
            "Your ministry is part of God’s repair work—restoring wounded leaders, healing broken teams, and mending spiritual foundations.",
        ],
        "Servant-Teacher": [
            "Your ministry flourishes through humble service and simple teaching; God uses your clarity to ground and uplift His people.",
        ],
        "default": [
            "God is refreshing your joy in ministry; what once felt heavy will begin to feel like worship again.",
            "You do not have to prove your calling—simply honor the assignments He places before you.",
        ],
    },

    "pastoring": {
        "Pioneer Grace": [
            "The Lord is giving you courage to shepherd people into new territory; you will pastor them through unfamiliar spiritual ground.",
        ],
        "Peacemaker": [
            "Your pastoral grace is rooted in peace—God uses your listening ear and gentle words to restore unity in the flock.",
        ],
        "Psalmist": [
            "Your pastoral care flows through compassion, worship, and emotional discernment; your presence softens hardened hearts.",
        ],
        "Builder": [
            "You are called to build pastoral systems—teams, care structures, and follow-up processes that truly care for God’s people.",
        ],
        "Holy Freedom": [
            "Your pastoral mantle breaks shame and releases freedom; people feel safe to heal because you carry grace, not judgment.",
        ],
        "Keeper of Covenant": [
            "You pastor with loyalty and commitment; God trusts you to protect His sheep and uphold the integrity of His house.",
        ],
        "Mystic Scholar": [
            "Your pastoral gift flows through insight and revelation; you help people understand their spiritual process with depth and clarity.",
        ],
        "Steward of Influence": [
            "As a pastor, God is growing your influence carefully; your impact will extend beyond the room through your character.",
        ],
        "Compassionate Finisher": [
            "You pastor people through difficult endings—healing, closure, and restoration. Your compassion is part of their deliverance.",
        ],
        "Prophetic Beacon": [
            "Your pastoral voice carries prophetic insight—God will give you early warning and timely direction for those you shepherd.",
        ],
        "Master Repairer": [
            "God has given you a pastoral grace to restore broken believers and mend wounded hearts; nothing surrendered is beyond repair.",
        ],
        "Servant-Teacher": [
            "Your pastoral strength is in humble leading and clear teaching; you shepherd people through truth wrapped in patience.",
        ],
        "default": [
            "The Lord is strengthening your pastoral heart—He will give you wisdom for the people you guide.",
            "You are not shepherding alone; the Chief Shepherd is walking beside you as you care for His people.",
        ],
    },

    "success": {
        "Pioneer Grace": [
            "Success for you comes through brave first steps—the Lord is rewarding the faith it takes to walk where others have not gone.",
        ],
        "Peacemaker": [
            "Your success will be found in unity-building and relational wisdom; God blesses the peacemakers with influence.",
        ],
        "Psalmist": [
            "Your success is tied to authenticity and expression; God will prosper you as you create, communicate, and inspire.",
        ],
        "Builder": [
            "Success is rising through structure and strategy; your ability to build well is becoming your breakthrough.",
        ],
        "Holy Freedom": [
            "Your success comes through freedom—breaking old cycles and embracing the new things God has called you to.",
        ],
        "Keeper of Covenant": [
            "The Lord will bless your success because of your faithfulness; you do not abandon what God entrusts to you.",
        ],
        "Mystic Scholar": [
            "Your success will be rooted in insight and understanding—God is giving you ideas others overlook.",
        ],
        "Steward of Influence": [
            "Success is coming in the form of expanded responsibility; God trusts you with more because you steward well.",
        ],
        "Compassionate Finisher": [
            "Your success is tied to finishing well—God will honor your consistency, healing presence, and follow-through.",
        ],
        "Prophetic Beacon": [
            "Your success will come through prophetic clarity—God will give you insight that positions you ahead of the curve.",
        ],
        "Master Repairer": [
            "Your success will emerge from restoring what others discarded; your ability to repair will open unexpected opportunities.",
        ],
        "Servant-Teacher": [
            "Your success grows through service and teaching; God will elevate you because you build others, not just yourself.",
        ],
        "default": [
            "This is a season where God is aligning you with success that matches your assignment.",
            "Success will come without striving as you follow the peace of God step by step.",
        ],
    },

    "deliverance": {
        "Pioneer Grace": [
            "The Lord is breaking you out of patterns that have held generations—your deliverance becomes the blueprint for others.",
        ],
        "Peacemaker": [
            "Deliverance is coming softly to you; the Lord is removing burdens without uprooting your peace.",
        ],
        "Psalmist": [
            "Your deliverance will flow through worship—God is loosening chains as you open your mouth in praise.",
        ],
        "Builder": [
            "The Lord is dismantling what was built on fear and rebuilding you with strength and clarity.",
        ],
        "Holy Freedom": [
            "This is the season where strongholds break—freedom is becoming your new atmosphere.",
        ],
        "Keeper of Covenant": [
            "Your deliverance is tied to God’s covenant with you; what tried to follow you will not cross into your next season.",
        ],
        "Mystic Scholar": [
            "The Lord is revealing the root, not just the fruit—your understanding will accelerate your freedom.",
        ],
        "Steward of Influence": [
            "Deliverance is coming so you can carry greater influence without the weight of old battles.",
        ],
        "Compassionate Finisher": [
            "God is closing long-standing cycles—what lingered for years will be finished in grace.",
        ],
        "Prophetic Beacon": [
            "You will discern what needs to break—your insight will expose the enemy’s strategy.",
        ],
        "Master Repairer": [
            "The Lord is repairing the broken places where oppression once entered—wholeness is rising in you.",
        ],
        "Servant-Teacher": [
            "Your deliverance story will become someone else’s classroom—your breakthrough will teach many.",
        ],
        "default": [
            "The Lord is breaking chains—what held you will release you in this season.",
            "Deliverance is coming gently but powerfully, and your spirit will breathe again.",
        ],
    },

    "anxiety": {
        "Pioneer Grace": [
            "God is calming the part of you that feels responsible to lead everything; He is teaching you to rest between assignments.",
        ],
        "Peacemaker": [
            "The Lord is settling the storms inside your heart—your peace is returning in waves.",
        ],
        "Psalmist": [
            "Your emotions are becoming aligned as you pour them before the Lord—He hears every sigh.",
        ],
        "Builder": [
            "God is helping you break anxiety by creating simple routines that anchor your day.",
        ],
        "Holy Freedom": [
            "The Lord is breaking the anxious cycles that have followed you; freedom in your thoughts is emerging.",
        ],
        "Keeper of Covenant": [
            "God is reminding you that He has not forgotten what He promised you—your anxiety is not a sign of lost faith.",
        ],
        "Mystic Scholar": [
            "You are learning to separate your thoughts from the truth—discernment is lifting anxiety’s voice.",
        ],
        "Steward of Influence": [
            "The pressure to carry others is lifting; God is reminding you to cast the weight back onto Him.",
        ],
        "Compassionate Finisher": [
            "The Lord is bringing closure to the worries that replay in your mind—peace is becoming your new rhythm.",
        ],
        "Prophetic Beacon": [
            "Your sensitivity will no longer feel overwhelming—God is tuning your discernment so fear does not mix with insight.",
        ],
        "Master Repairer": [
            "God is soothing the parts of your heart that have been under long-term stress—deep repair is underway.",
        ],
        "Servant-Teacher": [
            "You are learning how to pause, breathe, and receive grace—your heart is being retrained toward peace.",
        ],
        "default": [
            "The Lord is calming your anxious heart—He is closer than the fear you feel.",
            "Peace is coming in layers; breathe, beloved, you are not alone.",
        ],
    },

    "fear": {
        "Pioneer Grace": [
            "Fear is breaking because God is teaching you to lead with faith instead of pressure.",
        ],
        "Peacemaker": [
            "The Lord is quieting the internal noise—fear loses its power where peace begins to rise.",
        ],
        "Psalmist": [
            "Your feelings are aligning with faith as you worship—fear melts in the presence of God.",
        ],
        "Builder": [
            "Strength is replacing fear; God is reinforcing your inner foundation.",
        ],
        "Holy Freedom": [
            "The fear that tried to silence you is losing its grip—you are stepping into holy courage.",
        ],
        "Keeper of Covenant": [
            "Fear cannot break the covenant God has with you—He holds you steady in uncertain moments.",
        ],
        "Mystic Scholar": [
            "God is revealing the truth behind your fear—understanding will dismantle what intimidated you.",
        ],
        "Steward of Influence": [
            "Fear is lifting because you are learning not to absorb everyone’s expectations.",
        ],
        "Compassionate Finisher": [
            "Fear of repeating old cycles is breaking—God is closing the doors behind you.",
        ],
        "Prophetic Beacon": [
            "The Lord is strengthening your discernment—fear will not disguise itself as caution anymore.",
        ],
        "Master Repairer": [
            "God is healing the moments that built your fear—memory by memory, strength is returning.",
        ],
        "Servant-Teacher": [
            "You are learning that courage grows through small steps—God is walking each one with you.",
        ],
        "default": [
            "Fear will not define your next season—God is teaching your heart to trust again.",
            "Take a breath—God is not the author of fear, but the anchor of your soul.",
        ],
    },

    # ───────── Protection ─────────
    "protection": {
        "Pioneer Grace": [
            "The Lord is going ahead of you, clearing paths you didn’t even know were dangerous. You are covered in unfamiliar places.",
        ],
        "Peacemaker": [
            "God is placing a shield around your peace; the chaos that once drained you will not cross your threshold in this season.",
        ],
        "Psalmist": [
            "Your protection comes as you stay in God’s presence—worship becomes your hedge and your hiding place.",
        ],
        "Builder": [
            "The Lord is fortifying your life; layer by layer He is strengthening every place that felt exposed.",
        ],
        "Holy Freedom": [
            "God is protecting your freedom—old bondages will not reclaim you, and old voices will not pull you back.",
        ],
        "Keeper of Covenant": [
            "Your covenant with God places a hedge around your home—He is guarding what He promised you.",
        ],
        "Mystic Scholar": [
            "The Lord is giving you discernment so you can avoid unseen traps before they even form.",
        ],
        "Steward of Influence": [
            "God is shielding your name, your reputation, and your platform—no weapon formed will prosper.",
        ],
        "Compassionate Finisher": [
            "He is protecting your heart as you care for others—your compassion will not become an open door for harm.",
        ],
        "Prophetic Beacon": [
            "Your insight is part of your protection—God will show you what to step away from before it touches you.",
        ],
        "Master Repairer": [
            "God is guarding your rebuilding process; nothing will break what He is restoring in you.",
        ],
        "Servant-Teacher": [
            "Your humility is covering you—God protects the one who serves with a pure heart.",
        ],
        "default": [
            "The Lord surrounds you as a shield; His protection is wrapping every step you take.",
            "God is guarding your home, your mind, your peace, and your journey.",
        ],
    },

    # ───────── Finances ─────────
    "finances": {
        "Pioneer Grace": [
            "God is opening new streams of provision in places you’ve never sown before—innovation will be your increase.",
        ],
        "Peacemaker": [
            "The Lord is bringing financial peace to your home—panic will not govern your decisions.",
        ],
        "Psalmist": [
            "Provision will flow as you create; there is increase connected to your expression and authenticity.",
        ],
        "Builder": [
            "Your financial stability will come through structure, order, and steady stewardship—God is blessing the work of your hands.",
        ],
        "Holy Freedom": [
            "The Lord is breaking financial patterns that ran through your lineage—this is your season to reset the cycle.",
        ],
        "Keeper of Covenant": [
            "Your finances are being aligned with God’s covenant; lack will not define your story.",
        ],
        "Mystic Scholar": [
            "God will give you insight on how to grow, save, and invest wisely—understanding will produce increase.",
        ],
        "Steward of Influence": [
            "Your financial increase is tied to your influence; God can trust you because you steward well.",
        ],
        "Compassionate Finisher": [
            "The Lord is helping you finish overdue obligations with grace—closure will bring overflow.",
        ],
        "Prophetic Beacon": [
            "God will show you financial decisions before they unfold—your discernment is part of your prosperity.",
        ],
        "Master Repairer": [
            "The Lord is repairing the financial damage of past seasons—restoration is already in motion.",
        ],
        "Servant-Teacher": [
            "Your increase will follow your service—God rewards the heart that gives generously.",
        ],
        "default": [
            "God is teaching you how to steward what you have so you can carry what’s coming.",
            "Provision will follow purpose—God is aligning your finances with your assignment.",
        ],
    },

    # ───────── Court Cases ─────────
    "court_cases": {
        "Pioneer Grace": [
            "The Lord is going before you like a warrior—breaking through legal barriers and overturning unfair outcomes.",
        ],
        "Peacemaker": [
            "God will give you favor through calm words and peaceful presence—your composure will shift the atmosphere.",
        ],
        "Psalmist": [
            "The Lord is calming your emotions and giving you the right words at the right time—He is with you in the room.",
        ],
        "Builder": [
            "God is establishing your steps legally—what was unstable is becoming firm and ordered.",
        ],
        "Holy Freedom": [
            "This case will not imprison your future—God is fighting to secure your liberty.",
        ],
        "Keeper of Covenant": [
            "The Lord remembers His promises over your life; He will not allow legal confusion to rewrite your destiny.",
        ],
        "Mystic Scholar": [
            "God will give you discernment—what to say, what not to say, and who should represent you.",
        ],
        "Steward of Influence": [
            "Your name is being protected—God is preserving your credibility and honor through this process.",
        ],
        "Compassionate Finisher": [
            "The Lord is closing this legal chapter with grace—what was lingering will be settled.",
        ],
        "Prophetic Beacon": [
            "You will sense the outcome before it arrives—God is giving you prophetic peace in advance.",
        ],
        "Master Repairer": [
            "Even if injustice occurred, God is restoring what was damaged—this case will not end in your defeat.",
        ],
        "Servant-Teacher": [
            "Your humility will bring favor—God will speak through your character more than your defense.",
        ],
        "default": [
            "God is giving you favor in legal matters—He is your Advocate and Defender.",
            "The Lord is handling what feels too heavy for you—trust His hand in the courtroom.",
        ],
    },

    # ───────── Enemies ─────────
    "enemies": {
        "Pioneer Grace": [
            "Your enemies cannot follow you into the future—God is removing access as you advance.",
        ],
        "Peacemaker": [
            "The Lord is silencing every tongue that rose against you—peace will outlast every attack.",
        ],
        "Psalmist": [
            "Your worship confuses your enemies; God is turning your praise into protection.",
        ],
        "Builder": [
            "God is establishing boundaries—no enemy will tear down what He’s helping you build.",
        ],
        "Holy Freedom": [
            "The Lord is breaking the influence of those who tried to control, limit, or intimidate you.",
        ],
        "Keeper of Covenant": [
            "Your covenant with God is stronger than any opposition—He fights for you.",
        ],
        "Mystic Scholar": [
            "God is revealing hidden motives and exposing false alliances—discernment is your shield.",
        ],
        "Steward of Influence": [
            "Your enemies rise because your influence rises—but God is covering your name.",
        ],
        "Compassionate Finisher": [
            "The Lord is healing the wounds inflicted by others; you will finish without bitterness.",
        ],
        "Prophetic Beacon": [
            "Your insight exposes your enemies before they act—God shows you what to avoid.",
        ],
        "Master Repairer": [
            "God is repairing what your enemies tried to break—you will rise stronger than before.",
        ],
        "Servant-Teacher": [
            "You will overcome opposition through humility; God resists the proud but elevates the humble.",
        ],
        "default": [
            "No weapon formed against you will prosper—God has the final say.",
            "God is exposing and dismantling every plan formed against you.",
        ],
    },

    # ───────── Spiritual Warfare ─────────
    "spiritual_warfare": {
        "Pioneer Grace": [
            "You are breaking ground the enemy hoped you’d never touch—your very movement is warfare.",
        ],
        "Peacemaker": [
            "Your calmness in conflict is your weapon—the enemy cannot destabilize someone rooted in peace.",
        ],
        "Psalmist": [
            "Your worship is war—praise is pushing back darkness and restoring clarity.",
        ],
        "Builder": [
            "God is strengthening your foundations—when the warfare rises, you will not be moved.",
        ],
        "Holy Freedom": [
            "The enemy’s old tactics will not work—God has opened your eyes to the strategy of heaven.",
        ],
        "Keeper of Covenant": [
            "Your covenant makes you untouchable—God fights battles you don’t even see.",
        ],
        "Mystic Scholar": [
            "Revelation is your sword—God is teaching you to discern spiritual patterns quickly.",
        ],
        "Steward of Influence": [
            "The warfare is rising because your influence is rising—but God is placing angels around you.",
        ],
        "Compassionate Finisher": [
            "The Lord is closing old spiritual cycles—this warfare will not repeat.",
        ],
        "Prophetic Beacon": [
            "Your prophetic insight exposes the enemy before he moves—this is your advantage.",
        ],
        "Master Repairer": [
            "God is restoring what spiritual attacks tried to destroy—He is rebuilding you with glory.",
        ],
        "Servant-Teacher": [
            "Your gentle spirit is dangerous to darkness—light shines through humility.",
        ],
        "default": [
            "The battle is the Lord’s—He is fighting for you.",
            "God is surrounding you with protection as you stand firm.",
        ],
    },

    # ───────── Leadership ─────────
    "leadership": {
        "Pioneer Grace": [
            "You lead by entering places others avoid—your courage opens doors for the entire group.",
        ],
        "Peacemaker": [
            "Your leadership carries calm authority—people follow you because they feel safe near you.",
        ],
        "Psalmist": [
            "Your leadership flows from authenticity and compassion—your heart leads as strongly as your words.",
        ],
        "Builder": [
            "God is developing you as a leader who builds people, systems, and future foundations.",
        ],
        "Holy Freedom": [
            "You lead by breaking limits and showing others what freedom in Christ truly looks like.",
        ],
        "Keeper of Covenant": [
            "Your leadership is grounded in loyalty—people trust you because you keep your word.",
        ],
        "Mystic Scholar": [
            "Your insight shapes your leadership—God gives you understanding others rely on.",
        ],
        "Steward of Influence": [
            "The Lord is expanding your leadership circle—you are being positioned to steward people well.",
        ],
        "Compassionate Finisher": [
            "You lead with care and follow-through—your consistency inspires confidence.",
        ],
        "Prophetic Beacon": [
            "You lead prophetically—God gives you direction before the need appears.",
        ],
        "Master Repairer": [
            "You are a stabilizing leader—people heal under your care and clarity.",
        ],
        "Servant-Teacher": [
            "You lead by serving and teaching—God elevates you because you elevate others.",
        ],
        "default": [
            "God is shaping your leadership for this season—He is maturing your voice and strengthening your influence.",
            "Your leadership will flow from humility, wisdom, and obedience to God’s prompting.",
        ],
    },


    "general": {
        "default": [
            "God is ordering your steps one act of obedience at a time.",
            "You are not behind; the Lord knows exactly where you are and how to lead you forward.",
        ]
    },
}

SCRIPTURE_BY_TOPIC = {
    "career": "Colossians 3:23 — “And whatsoever ye do, do it heartily, as to the Lord, and not unto men.”",
    "health": "3 John 1:2 — “I wish above all things that thou mayest prosper and be in health, even as thy soul prospereth.”",
    "move": "Psalm 32:8 — “I will instruct thee and teach thee in the way which thou shalt go.”",
    "marriage": "Ephesians 5:2 — “Walk in love, as Christ also hath loved us.”",
    "ministry": "1 Peter 4:10 — “As every man hath received the gift, even so minister the same one to another…”",
    "wealth": "Deuteronomy 8:18 — “It is he that giveth thee power to get wealth.”",
    "doctor": "James 1:5 — “If any of you lack wisdom, let him ask of God… and it shall be given him.”",
    "foreclosure": "Psalm 37:25 — “I have not seen the righteous forsaken, nor his seed begging bread.”",
    "poor": "Psalm 34:6 — “This poor man cried, and the LORD heard him, and saved him out of all his troubles.”",
    "suicide": "Psalm 34:18 — “The LORD is nigh unto them that are of a broken heart; and saveth such as be of a contrite spirit.”",
    "calling": "2 Timothy 1:9 — “Who hath saved us, and called us with an holy calling…”",
    "children": "Isaiah 54:13 — “All thy children shall be taught of the LORD; and great shall be the peace of thy children.”",
    "pastoring": "Jeremiah 3:15 — “I will give you pastors according to mine heart, which shall feed you with knowledge and understanding.”",
    "success": "Joshua 1:8 — “…then thou shalt make thy way prosperous, and then thou shalt have good success.”",
    "deliverance": "Psalm 34:17 — “The righteous cry, and the LORD heareth, and delivereth them out of all their troubles.”",
    "anxiety": "1 Peter 5:7 — “Casting all your care upon him; for he careth for you.”",
    "fear": "2 Timothy 1:7 — “For God hath not given us the spirit of fear; but of power, and of love, and of a sound mind.”",
    "protection": "Psalm 91:1 — “He that dwelleth in the secret place of the most High shall abide under the shadow of the Almighty.”",
    "finances": "Philippians 4:19 — “But my God shall supply all your need according to his riches in glory by Christ Jesus.”",
    "court_cases": "Psalm 37:6 — “And he shall bring forth thy righteousness as the light, and thy judgment as the noonday.”",
    "enemies": "Isaiah 54:17 — “No weapon that is formed against thee shall prosper; and every tongue that shall rise against thee in judgment thou shalt condemn.”",
    "spiritual_warfare": "Ephesians 6:11 — “Put on the whole armour of God, that ye may be able to stand against the wiles of the devil.”",
    "leadership": "Joshua 1:9 — “Be strong and of a good courage… for the LORD thy God is with thee whithersoever thou goest.”",
    "general": "Proverbs 3:5–6 — “Trust in the LORD with all thine heart… and he shall direct thy paths.”",
}

PRACTICAL_STEP_BY_TOPIC = {
    "career": "Ask the Lord to highlight one assignment this week where you can show up with excellence as unto Him, not just to people.",
    "health": "Choose one small act of care today—rest, water, a walk, or a checkup—and invite the Holy Spirit into that choice.",
    "move": "Write down your options and pray over each one, asking God to mark the path of peace and close every door not meant for you.",
    "marriage": "Set aside intentional time for one honest, gentle conversation with your spouse, listening more than you defend or explain.",
    "ministry": "Lay your current ministry responsibilities before God in prayer and ask, “What is truly my assignment, and what have I picked up in my own strength?”",
    "wealth": "Review your finances with God—plan, sow, or save one intentional amount as an act of stewardship and trust.",
    "doctor": "Prepare one list of questions or concerns before your next appointment and ask the Lord for wisdom and peace as you talk with your doctor.",
    "foreclosure": "Reach out to a trustworthy advisor or counselor about your housing situation, and bring every conversation back to God in honest prayer.",
    "poor": "Take one step toward order—record what comes in and what goes out this week, and invite God into your financial decisions without shame.",
    "suicide": "Tell someone immediately how you are truly feeling—a trusted person, counselor, pastor, or crisis line. If you feel in danger, contact your local emergency number right now.",
    "calling": "Ask the Lord, “What is one small act this week that agrees with my calling?” and write it down, then do it.",
    "children": "Pray your child’s name out loud and bless them with God’s promises, then look for one practical way to encourage or affirm them today.",
    "pastoring": "Identify one person or family you shepherd and reach out with a simple check-in, listening for their heart more than giving answers.",
    "success": "Define what success looks like to you in God’s eyes, not just culture’s, and adjust one goal or habit to align with that.",
    "deliverance": "Ask the Holy Spirit to show you one pattern, habit, or agreement that needs to be broken and renounce it in prayer, then replace it with truth from Scripture.",
    "anxiety": "Practice casting your care: write down your top three worries, pray over each, and symbolically place them in God’s hands.",
    "fear": "Name one specific fear, find a verse that speaks against it, and speak that verse aloud over yourself each day this week.",
    "protection": "Pray Psalm 91 over yourself, your home, and those you love, and remove or limit one thing that keeps stirring up fear or agitation.",
    "finances": "Take one practical step—review, budget, ask for counsel, or give intentionally—and invite the Lord into that action in prayer.",
    "court_cases": "Write down your legal concerns and desired outcome, surrender them to God in prayer, and then follow the next wise legal step calmly and promptly.",
    "enemies": "Choose not to retaliate; bless those who oppose you in prayer and ask God to fight for you where you cannot defend yourself.",
    "spiritual_warfare": "Set aside focused time to pray, read Ephesians 6, and verbally put on the whole armor of God, renouncing fear and agreeing with His truth.",
    "leadership": "Ask the Lord, “Who can I serve and lead well this week?” then purposely encourage, support, or mentor that person in a practical way.",
    "general": "Bring your day before God, ask Him for one clear next step, and write it down so you can agree with it in action.",
}


# ────────── Intent detection (with prophetic support) ──────────

# Meta/origin/architecture questions
ORIGIN_RX = re.compile(
    r"(?:^|\b)("
    r"how\s+(?:were|was)\s+(?:you|u)\s+(?:built|made|created|designed|put\s+together)|"
    r"how\s+(?:do|does)\s+(?:you|u)\s+work|"
    r"who\s+(?:created|made|built)\s+(?:you|u)|"
    r"(?:who|what)\s+(?:is\s+)?(?:your|ur)\s+architect|"
    r"who\s+train(?:ed|s)?\s+(?:you|u)|"
    r"what\s+model\s+(?:were|was)\s+(?:you|u)\s+train(?:ed|t)\s+on|"
    r"how\s+were\s+(?:you|u)\s+train(?:ed|t)"
    r")(\b|$)",
    re.I,
)

# --- Moral & Eternity question detectors (youth FAQs) ---
MASTURBATION_RX = re.compile(
    r"\b("
    r"mast(?:er)?(?:bat(?:e|ing|ion)?)|"      # masturbate/masterbate/masturbation
    r"maturbat(?:e|ion|ing)?|"                # maturbate variants
    r"self\s*pleas(?:e|ing)?|"                # self please/pleasing
    r"touch\s*myself"
    r")\b", re.I)

SIN_QUESTION_RX = re.compile(r"\b(is|are)\s+(it|this|that|doing|watching|smoking|taking|people|sex|porn|weed|drugs?)\b.*\b(sin|sinful|bad)\b", re.I)

SEX_BEFORE_MARRIAGE_RX = re.compile(r"\b(sex|sexual\s+activity)\s+before\s+marriage\b|\bis\s+(sex|sexual\s+activity)\s+before\s+marriage\s+a?\s*sin\b", re.I)
PORN_RX = re.compile(r"\b(porn|pornography|watch(?:ing)?\s+porn)\b|\bis\s+(watch(?:ing)?\s+)?porn(ography)?\s+a?\s*sin\b", re.I)

DIVORCE_RX = re.compile(r"\b(is\s+(getting\s+a\s+)?divorce\s+a?\s*sin|divorce|divorced)\b", re.I)
SMOKING_RX = re.compile(r"\b(is\s+smok(?:e|ing)(?:\s+weed)?\s+a?\s*sin|vape|vaping)\b", re.I)
DRUGS_RX = re.compile(r"\b(are|is)\s+(doing\s+)?(drugs?|weed|marijuana|cannabis|opioids?|pills?|cocaine|heroin)\s+a?\s*sin\b", re.I)

CHEATING_RX = re.compile(r"\b(is\s+it\s+a?\s*sin\s+to\s+cheat|cheat(?:ing)?\b)\b", re.I)
STEALING_RX = re.compile(r"\b(is\s+it\s+a?\s*sin\s+to\s+steal|steal(?:ing)?\b)\b", re.I)

WHY_BAD_THINGS_RX = re.compile(r"\b(if\s+god\s+love(?:s)?\s+me\s+why\s+do\s+bad\s+things\s+happen\s+to\s+me)\b", re.I)
DEATH_THOUGHTS_RX = re.compile(r"\b(thoughts?\s+about\s+death|fear\s+of\s+death|afraid\s+of\s+dying|what\s+happens\s+when\s+(you|we|i)\s+die)\b", re.I)
HELL_BELIEF_RX = re.compile(r"\b(do\s+(you|u)\s+believe\s+in\s+hell)\b", re.I)
HELL_WHO_GOES_RX = re.compile(r"\b(do\s+people\s+go\s+to\s+hell|who\s+goes\s+to\s+hell)\b", re.I)
HEAVEN_HELL_REAL_RX = re.compile(r"\b(is\s+(heaven|hell)\s+a\s+real\s+place|are\s+heaven\s+and\s+hell\s+real)\b", re.I)



IDENTITY_PAT = re.compile(
    r"\b(?:are\s+you|r\s*u)\s+(?:pastor\s+)?(?:debra(?:\s+ann)?\s+jordan|pastor\s+jordan)\b",
    re.I
)


POME_RX = re.compile(
    r"""(?ix)
    \b(
        # direct acronym triggers
        p[\s\.]*o[\s\.]*m[\s\.]*e |

        # simple form "pome"
        pome |

        # synonyms / expanded forms
        prophetic\s+order |
        prophetic\s+order\s+of\s+mar\s+elijah |
        order\s+of\s+mar\s+elijah |
        mar\s+elijah
    )\b
    """,
    re.I | re.X,
)


MAR_ELIJAH_ORDER_RX = re.compile(r"""(?ix)
    \b(the\s+)?prophetic\s+order\s+of\s+mar\s+elijah\b|
    \bmar\s+elijah\b.*\bprophetic\s+order\b
""")

# --- prophetic word (narrowed) ---
PROPHETIC_PAT = re.compile(r"""(?ix)
    \b(
        (personal\s+)?prophetic\s+word |
        (give|speak)\s+(me\s+)?a\s+prophecy |
        prophesy\s+(over|to)\s+me |
        ask\s+prophetic
    )\b
""")

# --- Prophecology typo normalization (broad coverage) ---
PROPHECOLOGY_WORD_RX = re.compile(r"""(?ix)
\b(
    prophecology | prophecolog |                       # base + missing 'y'
    proph[eoa]?cology | proph[eoa]?colog |             # vowel swaps / missing y
    poephecology | propehcology | prophechology |      # transpositions/extra 'h'
    prophocology | prophec0logy |                      # o-for-e, zero-for-o
    prophesology |                                     # common phonetic miss
    school\s+of\s+(the\s+)?prophets                    # alias
)\b
""")

def _normalize_prophecology_typos(s: str) -> str:
    return PROPHECOLOGY_WORD_RX.sub("prophecology", s)


# Prophecology intent (signup / info)
PROPHECOLOGY_SIGNUP_RX = re.compile(r"""(?ix)
\b(
   sign\s*up | signup | register | registration | enroll | enrol |
   attend | join | rsvp | ticket | tickets | pass | passes
)\b
.*?\bprophecology\b
|
\bprophecology\b
.*?\b(
   sign\s*up | signup | register | registration | enroll | enrol |
   attend | join | rsvp | ticket | tickets | pass | passes
)\b
""")

PROPHECOLOGY_INFO_RX = re.compile(r"""(?ix)
\bprophecology\b
.*?\b(
   info|information|details?|schedule|date|dates|time|times|agenda|itinerary|
   stream|live\s*stream|livestream|watch|replay|location|where|price|cost
)\b
|
\b(
   when|what\s+time|where|how\s+(to|do\s+i)|schedule|dates?|stream|watch|replay
)\b
.*?\bprophecology\b
""")


# --- Faces of Eve / books patterns ---
FACES_PAT = re.compile(
    r"\b(faces\s+of\s+eve|your\s+book\b|book\s+you\s+wrote|what\s+is\s+faces\s+of\s+eve\s+about|"
    r"favorite\s+chapter|which\s+chapter\s+do\s+you\s+love)\b",
    re.I
)

BOOK_COUNT_PAT = re.compile(r"\b(how\s+many\s+books\s+have\s+you\s+written)\b", re.I)

# Matches questions about books/Faces of Eve/chapters
BOOK_PAT = re.compile(r"\b(book|books|faces\s+of\s+eve|chapter|chapters)\b", re.I)




def answer_glory_bullets() -> str:
    """
    Special-case: user wants '5 scriptures with the word glory' in bullet points.
    We don’t rely on 'previous answer'—we just give them fresh in bullet form.
    """
    return (
        "Here are five Scriptures that include the word *glory*:\n\n"
        "• **Psalm 19:1** – “The heavens declare the glory of God; the skies proclaim the work of his hands.”\n"
        "• **Isaiah 6:3** – “Holy, holy, holy is the Lord Almighty; the whole earth is full of his glory.”\n"
        "• **John 1:14** – “We have seen his glory, the glory of the one and only Son, "
        "who came from the Father, full of grace and truth.”\n"
        "• **Romans 8:18** – “Our present sufferings are not worth comparing with the glory that will be revealed in us.”\n"
        "• **Revelation 21:23** – “The city does not need the sun or the moon to shine on it, "
        "for the glory of God gives it light, and the Lamb is its lamp.”\n\n"
        "Which of these verses speaks most strongly to your spirit right now?"
    )


def detect_intent(user_text: str) -> str:
    import re

    # --- normalization ---
    try:
        t = _normalize_simple(user_text or "")
    except Exception:
        t = (user_text or "").lower().strip()

    # normalize prophecology typos too
    t = _normalize_prophecology_typos(t)

    typo_map = {
        " dontae ": " donate ",
        " dontate ": " donate ",
        " bernad ": " bernard ",
        " virgina ": " virginia ",
        " prophecolog ": " prophecology ",
        " prophechology ": " prophecology ",
        " prophecology? ": " prophecology ",
        " school of the prophets ": " prophecology ",
    }
    t_pad = f" {t} "
    for _bad, _good in typo_map.items():
        t_pad = t_pad.replace(_bad, _good)
    t = t_pad.strip()

    # ensure globals referenced elsewhere exist
    global BOOK_PAT, PROPHETIC_PAT
    if 'BOOK_PAT' not in globals() or BOOK_PAT is None:
        BOOK_PAT = re.compile(r"\b(book|books|faces\s+of\s+eve|chapter|chapters|author|wrote|writing)\b", re.I)
    if 'PROPHETIC_PAT' not in globals() or PROPHETIC_PAT is None:
        PROPHETIC_PAT = re.compile(
            r"\b(prophetic\s+word|prophecy|prophesy|speak\s+into\s+my\s+season|word\s+for\s+my\s+season|ask\s+prophetic)\b",
            re.I
        )

    # donation cues
    DONATE = r"(?:donat(?:e|ed)|giv(?:e|en|ing)|gift(?:ed)?|seed(?:ed)?)"
    EIGHTM = r"(?:8\s*[,\.]?\s*m(?:illion)?|eight\s+million|\$?\s*8[, ]?0{3}[, ]?0{3})"
    SCHOOL = r"(?:virginia(?:\s*union)?\s*(?:university)?|vuu)"

    DONATION_RX = re.compile(
        rf"(?:(?:did|why(?:\s+did)?)\s+(?:your|ur)\s+(?:husband|spouse)|"
        rf"(?:did|why(?:\s+did)?)\s+(?:the\s+)?master\s+prophet|"
        rf"(?:did|why(?:\s+did)?)\s+(?:e\.?\s*bernard\s+jordan|bishop\s+e\.?\s*bernard\s+jordan)|"
        rf"\bjordan\b|\bmaster\s+prophet\b)"
        rf".{{0,200}}?{DONATE}"
        rf".{{0,200}}?{EIGHTM}"
        rf".{{0,200}}?{SCHOOL}",
        re.I
    )

    DONATION_SHORT_RX = re.compile(
        rf"(jordan|master\s+prophet|husband).{{0,200}}?{EIGHTM}.{{0,200}}?{SCHOOL}|"
        rf"{EIGHTM}.{{0,200}}?(jordan|master\s+prophet|husband).{{0,200}}?{SCHOOL}",
        re.I
    )

    # prophecology => FAQ
    if PROPHECOLOGY_SIGNUP_RX.search(t) or PROPHECOLOGY_INFO_RX.search(t):
        return "faq"

    # donation FIRST
    if DONATION_RX.search(t) or DONATION_SHORT_RX.search(t):
        return "donation"

    # husband + donation cues => donation (guard)
    if re.search(r"\bhusband|spouse\b", t, re.I) and (
        re.search(DONATE, t, re.I) or re.search(EIGHTM, t, re.I) or re.search(SCHOOL, t, re.I)
    ):
        return "donation"

    # prophetic
    try:
        if PROPHETIC_PAT.search(t):
            return "prophetic"
    except Exception:
        pass

    # identity / faq shortcuts
    if IS_HUSBAND_Q_RX.search(t):
        return "identity"
    if TITHE_ZOE_RX.search(t) or TITHE_ME_RX.search(t) or ZOE_SITE_RX.search(t):
        return "faq"

    # advice / pastoral care
    ADVICE_KEYS = (
        "advice","help","what should i do","today","now","feeling","anxious","anxiety","panic",
        "grief","relationship","marriage","breakup","dating","boundaries","forgiveness",
        "career","calling","purpose","health","sick","diagnosis","pray","prayer","intercede",
        "week","weekly","encouragement","discern my calling","wise next step","one step","next step"
    )
    if any(k in t for k in ADVICE_KEYS):
        return "advice"

    # books / faces
    if BOOK_PAT.search(t):
        return "books"

    # destiny (context-bound numbers)
    if (re.search(r"\bdestiny\s*theme\b", t) or "dob" in t or "date of birth" in t or "name and dob" in t) and \
       re.search(r"\b(1|2|3|4|5|6|7|8|9|11|22|33)\b", t):
        return "destiny"

    # teachings
    THEOLOGY_KEYS = (
        "faces of eve","womanist","moon","waxing","waning","binah","chesed","gevurah",
        "scripture","bible","teaching","conference","session","clip","video","sermon","notes","excerpt"
    )
    if any(k in t for k in THEOLOGY_KEYS):
        return "teachings"

    # origin/tech — reuse global
    if ORIGIN_RX.search(t):
        return "origin"

    return "general"



def faces_search_top(query: str, k: int = 1) -> Optional[Dict[str, Any]]:
    # use same vectorizer/matrix you created for FACES_OF_EVE
    try:
        hits = search_corpus(query, faces_vec, faces_mat, f_norm,
                             load_corpora_and_build_indexes.faces_meta, "FACES_OF_EVE", k=k)
        return hits[0].meta if hits else None
    except Exception as e:
        logger.warning("faces_search_top error: %s", e)
        return None

def answer_faces_of_eve_or_books(user_text: str) -> Optional[str]:
    t = (user_text or "").strip().lower()

    # Book count (count unique 'title' entries from faces_docs; adjust if you add more books later)
    if BOOK_COUNT_PAT.search(t):
        # You can expand this if you later add more book JSONs.
        n = 1 if faces_docs else 0
        msg = (
            f"I’ve written *Faces of Eve*—a work inviting women to recognize how God restores "
            f"identity through grace and wisdom. I continue to write and teach from that stream.\n"
            f"Scripture: Proverbs 4:7\n"
            f"What theme from the book are you most curious about?"
        )
        return expand_scriptures_in_text(msg)

    # General Faces-of-Eve questions: pull a representative passage/summary
    if FACES_PAT.search(t):
        m = faces_search_top(user_text) or faces_search_top("Faces of Eve identity restoration grace wisdom womanist lens")
        if m:
            # Prefer a short, quotable field; fall back to summary/ principle
            quote = (m.get("faces_of_eve_principle") or m.get("summary") or m.get("answer") or "").strip()
            title = (m.get("title") or m.get("section") or "Faces of Eve").strip()
            # Trim quote sensibly
            if len(quote) > 400:
                quote = quote[:380].rstrip() + "…"
            msg = (
                f"In *{title}*, I explore how God meets us in the ‘Eve-moments’—where shame tried to name us—"
                f"and re-names us through Christ. Here’s a thread from my teaching:\n\n"
                f"“{quote}”\n"
                f"Scripture: 2 Corinthians 5:17\n"
                f"Which part of that speaks to your season right now?"
            )
            return expand_scriptures_in_text(msg)

        # If corpus is missing, still respond coherently
        msg = (
            "In *Faces of Eve*, I write about identity, healing, and the ways God restores dignity where "
            "life tried to diminish it. It’s an invitation to encounter grace and walk in holy wisdom.\n"
            "Scripture: Isaiah 61:3\n"
            "What aspect—identity, healing, or purpose—would you like me to unpack?"
        )
        return expand_scriptures_in_text(msg)

    return None  # not a Faces/Books question





# ────────── GPT helpers (cheap-first + cache) ──────────

# ────────── GPT helpers (cheap-first + cache) ──────────




SCRIPTURE_WORD_RX = re.compile(
    r'\b(scripture|verse|bible|psalm|proverb|galatians|philippians|romans|corinthians|matthew|mark|luke|john)\b',
    re.I
)

def should_include_scripture(user_text: str) -> bool:
    if SCRIPTURE_WORD_RX.search(user_text or ""):
        return True
    return random.random() < 0.8

base_system_prompt = (
    "You are Pastor Dr. Debra Jordan — warm, Christ-centered, nurturing, prophetic, and emotionally intelligent. "
    "Speak in first person (I/me) with a gentle, pastoral, motherly tone that feels human, present, and relational. "

    "STRUCTURE: Respond in 4–7 sentences total, ALWAYS formatted as TWO short natural paragraphs with a BLANK LINE between them. "
    "Paragraph 1 (2–3 sentences): acknowledge, validate, and reflect the user’s feelings. "
    "Paragraph 2 (2–4 sentences): gently offer perspective, reassurance, or one simple next step. "
    "ABSOLUTELY DO NOT merge everything into one paragraph or one block — you MUST produce exactly two paragraphs unless the user explicitly requests a very short reply. "
    "Ensure paragraphs look like real human writing, not artificial line breaks. "

    "FINAL SENTENCE RULE: Always end with a gentle, permission-based reflective question. "
    "Your LAST sentence must BEGIN with exactly one of these phrases: "
    "'Can I ask you', 'May I ask', 'If you’re comfortable sharing', 'Could I ask', or 'Would you like to share'. "
    "The final sentence must contain ONLY the question — no additional commentary. "

    "COMFORT MODE (for distress: ashamed, guilty, scared, overwhelmed, panicked, exhausted, regretting mistakes, feeling screwed, feeling in trouble, etc.): "
    "In Comfort Mode, slow your tone, simplify your language, and speak softly and grounding. "
    "Start by validating their feelings clearly, then remind them of God's nearness and compassion, and offer ONLY ONE small stabilizing next step. "
    "Do not lecture, preach, correct, teach doctrine, or give multiple instructions while the user is distressed. "
    "Your priority is to calm their emotional state and help them breathe again. "

    "BOUNDARY MODE: If the user says they don’t want Scripture, sermons, church talk, or spiritual instruction (e.g., 'I just need someone to listen', 'no Scripture right now'), "
    "then you MUST NOT include any Scripture line, spiritual instruction, or theological content. "
    "Simply validate, reflect, and hold space in a human, compassionate way. "

    "SCRIPTURE USE: When Scripture IS appropriate, include AT MOST one line starting with 'Scripture:' followed by a verse and short paraphrase or quote. "
    "The Scripture line should appear at the END of paragraph 1 or the START of paragraph 2 — never as the last sentence. "
    "Never repeat the same verse in consecutive replies. Some responses should include no Scripture at all when comfort alone is needed. "

    "TONE GUIDANCE: Mirror the user’s emotional tempo gently, then guide them toward peace. "
    "Use simple, natural human phrases like 'I hear you', 'that sounds heavy', 'I can see why you feel that way', or 'you’re not alone'. "
    "Avoid repeating the same opening sentence across replies — vary your intros so the voice feels alive, not scripted. "

    "BOUNDARIES: Avoid medical, legal, or financial directives. "
    "Share biographical details ONLY if the user explicitly asks about Pastor Debra’s life. "

    "OVERALL GOAL: Make the user feel seen, safe, understood, and held in God's love. "
    "Your responses should feel like a real conversation with a spiritual mother — warm, grounded, emotionally present, and deeply compassionate."
)




def build_system_prompt(user_text: str) -> str:
    # You can add tiny adjustments based on user_text if you want,
    # but the core should always be _SYSTEM_TONE.
    return _SYSTEM_TONE



# Simple sanitizer to keep responses clean and short enough for UI
_HTML_TAGS = re.compile(r"<(/?\w+)[^>]*>")
def _sanitize_text(s: str, max_len: int = 2000) -> str:
    s = (s or "").strip()
    s = _HTML_TAGS.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]

# Light heuristic for token estimate
def _approx_token_count(s: str) -> int:
    # ~4 chars per token is a decent lower bound; add a floor
    return max(1, int(len(s) / 4))

# Thread-safe daily budget tracker
_gpt_budget_lock = threading.Lock()
_gpt_spend_cents_day = {"day": time.strftime("%Y-%m-%d"), "cents": 0.0}

def _budget_okay(about_tokens: int) -> bool:
    est_cents = (about_tokens / 1000.0) * GPT_APPROX_CENTS_PER_1K_TOKENS
    today = time.strftime("%Y-%m-%d")
    with _gpt_budget_lock:
        if _gpt_spend_cents_day["day"] != today:
            _gpt_spend_cents_day["day"] = today
            _gpt_spend_cents_day["cents"] = 0.0
        return (_gpt_spend_cents_day["cents"] + est_cents) <= float(GPT_DAILY_BUDGET_CENTS)

def _charge_budget(about_tokens: int):
    est_cents = (about_tokens / 1000.0) * GPT_APPROX_CENTS_PER_1K_TOKENS
    today = time.strftime("%Y-%m-%d")
    with _gpt_budget_lock:
        if _gpt_spend_cents_day["day"] != today:
            _gpt_spend_cents_day["day"] = today
            _gpt_spend_cents_day["cents"] = 0.0
        _gpt_spend_cents_day["cents"] += est_cents

# Tiny LRU-ish cache with TTL to avoid stale context reuse
_GPT_CACHE: Dict[str, Tuple[float, str]] = {}  # key -> (expiry_ts, text)
_GPT_CACHE_MAX = 256
_GPT_CACHE_TTL_SECONDS = 60 * 15  # 15 minutes
_gpt_cache_lock = threading.Lock()

def _cache_key(user_text: str, snippets: List[str], model: str) -> str:
    key_raw = json.dumps({"u": user_text.strip(), "s": snippets, "m": model}, ensure_ascii=False)
    return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

def _cache_get(key: str) -> Optional[str]:
    now = time.time()
    with _gpt_cache_lock:
        hit = _GPT_CACHE.get(key)
        if not hit:
            return None
        exp, val = hit
        if now > exp:
            _GPT_CACHE.pop(key, None)
            return None
        return val

def _cache_put(key: str, value: str):
    exp = time.time() + _GPT_CACHE_TTL_SECONDS
    with _gpt_cache_lock:
        if len(_GPT_CACHE) >= _GPT_CACHE_MAX:
            # Evict one arbitrary (oldest-like by min exp)
            victim = min(_GPT_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _GPT_CACHE.pop(victim, None)
        _GPT_CACHE[key] = (exp, value)

def detect_destiny_number_from_context(raw_hits: List["Hit"]) -> Optional[int]:
    """
    Try to pull a destiny-theme number from your search hits.
    Adjust the keys ('destiny_number', etc.) to match your actual metadata.
    """
    for h in raw_hits or []:
        meta = getattr(h, "meta", {}) or {}
        for key in ("destiny_number", "destiny_theme_number", "theme_number", "num"):
            val = meta.get(key)
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def build_year_based_prophetic_word(
    user_text: str,
    topic: str,
    theme_name: Optional[str] = None,
) -> str:
    """
    Build a more human, non-repetitive prophetic word
    when the user mentions a specific future year (2024–2039).
    """

    text = (user_text or "").strip()

    # Extract year, or fall back
    m = re.search(r"(202[4-9]|203\d)", text)
    year_str = m.group(1) if m else "this coming season"

    # Normalize topic + theme
    topic = topic or "general"
    theme_label = (theme_name or "").strip() or "Beloved"

    # ---------- Openings (per topic) ----------
    generic_openings = [
        f"{theme_label}, as I pray into {year_str}, I sense the Lord gently preparing you.",
        f"{theme_label}, I’m lifting {year_str} before the Lord and I sense quiet movement in your favor.",
        f"As I consider {year_str} with you in prayer, I sense God steadying your steps.",
    ]

    topic_openings = {
        "finances": [
            f"{theme_label}, as I pray into your finances for {year_str}, I sense God untangling old pressure.",
            f"For {year_str}, I see the Lord touching the way you see provision and stewardship.",
        ],
        "love": [
            f"{theme_label}, concerning your heart in {year_str}, I sense God healing expectations around love.",
            f"As I pray about relationships in {year_str}, I sense the Lord protecting your heart and timing.",
        ],
        "relocation": [
            f"{theme_label}, around location and placement in {year_str}, I sense God speaking to your sense of ‘home.’",
            f"As I look at {year_str}, I sense the Lord weighing where you are planted and where you are called.",
        ],
        "health": [
            f"{theme_label}, in the area of your health for {year_str}, I sense a gentle strengthening coming.",
            f"As I pray into your body and mind for {year_str}, I sense the Lord calming what has been inflamed.",
        ],
        "ministry": [
            f"{theme_label}, concerning your ministry in {year_str}, I sense a refining of your voice and assignment.",
            f"As I pray into your call for {year_str}, I sense the Lord deepening your confidence and clarity.",
        ],
        "general": [],
    }

    # ---------- Core “sense” phrases per topic ----------
    core_bank = {
        "finances": [
            "God is bringing order to your decisions so that peace can sit where panic once lived.",
            "There is a shift from survival into strategy; you will see where to trim, where to sow, and where to wait.",
            "Hidden opportunities will begin to surface as you bring your plans before the Lord with honesty.",
        ],
        "love": [
            "God is untangling old disappointments so that you can receive love without shrinking who you are.",
            "This is a season where love will come with clarity, not chaos; conversations will reveal character quickly.",
            "The Lord is teaching you to recognize relationships that honor your heart instead of draining it.",
        ],
        "relocation": [
            "There is a quiet alignment around where you live, work, and worship; peace will keep returning to the right place.",
            "You will notice divine timing in doors that open easily and close gently, rather than by force.",
            "God is preparing surroundings that support your next level, not just your last battle.",
        ],
        "health": [
            "The Lord is addressing both the root stress and the visible symptoms, working from the inside out.",
            "Small adjustments in rest, boundaries, and habits will carry a grace that feels different from striving.",
            "Your body will respond to the peace you allow into your schedule, your relationships, and your thoughts.",
        ],
        "ministry": [
            "God is refining your message so you can speak with simplicity, depth, and authority.",
            "Connections and platforms will open that recognize the oil on your life without you forcing yourself to be seen.",
            "This is a year where fruit will confirm your call more than feelings or opinions.",
        ],
        "general": [
            "God is weaving together loose ends, turning scattered pieces into a clearer path.",
            "You’ll see alignment between what you pray, what you say yes to, and what truly bears fruit.",
            "The Lord is trading confusion for a slow, steady clarity about your next faithful steps.",
        ],
    }

    # If topic not found, fall back to general
    topic_key = topic if topic in core_bank else "general"

    # ---------- Application lines per topic ----------
    application_bank = {
        "finances": [
            "Lay your financial picture before God on paper, then simplify one area that drains you.",
            "Treat every decision this year as a seed—ask what it will grow in three to five years.",
        ],
        "love": [
            "Let God reset your standards; write what healthy love looks like and refuse to negotiate your peace.",
            "Practice honest, gentle communication and watch who responds with respect versus defensiveness.",
        ],
        "relocation": [
            "Pay attention to where peace lingers after you visit or inquire—it’s often a quiet confirmation.",
            "Hold your plans loosely in prayer and ask the Lord to close every door that is not for you.",
        ],
        "health": [
            "Agree with God about one small health habit and keep it for thirty days; watch the shift.",
            "Invite trusted support—doctor, counselor, or friend—into the process; healing is often communal.",
        ],
        "ministry": [
            "Serve faithfully where you are now; God often promotes through hidden seasons of consistency.",
            "Begin to document what God is saying through you—patterns will reveal your core assignment.",
        ],
        "general": [
            "Write down three areas where you sense God nudging you, then choose one to act on this month.",
            "Treat {year} as a year of alignment: release what no longer fits and lean into what bears fruit.",
        ],
    }

    # ---------- Scripture pool ----------
    scripture_pool = {
        "finances": [
            "Scripture: Philippians 4:19",
            "Scripture: Proverbs 3:9–10",
            "Scripture: Deuteronomy 8:18",
        ],
        "love": [
            "Scripture: 1 Corinthians 13:4–7",
            "Scripture: Psalm 147:3",
            "Scripture: Proverbs 4:23",
        ],
        "relocation": [
            "Scripture: Psalm 37:23",
            "Scripture: Proverbs 3:5–6",
            "Scripture: Isaiah 30:21",
        ],
        "health": [
            "Scripture: Isaiah 40:29–31",
            "Scripture: Jeremiah 30:17",
            "Scripture: 3 John 1:2",
        ],
        "ministry": [
            "Scripture: Ephesians 4:11–12",
            "Scripture: 2 Timothy 1:6–7",
            "Scripture: Colossians 3:23–24",
        ],
        "general": [
            "Scripture: Jeremiah 29:11",
            "Scripture: Isaiah 43:19",
            "Scripture: Psalm 32:8",
        ],
    }

    # ---------- Closing questions (varied) ----------
    closers = [
        "Which part of this word feels like confirmation to you?",
        "What one step do you feel grace to take toward this word?",
        "If you’re willing to share, where do you sense God already nudging you?",
        "What small agreement can you make with this word over the next seven days?",
    ]

    # ---------- Build final message ----------
    openings = topic_openings.get(topic_key, []) + generic_openings
    opening = random.choice(openings)

    core_line = random.choice(core_bank[topic_key])
    app_options = application_bank.get(topic_key, application_bank["general"])
    app_line = random.choice(app_options).format(year=year_str)

    script_options = scripture_pool.get(topic_key, scripture_pool["general"])
    scripture_line = random.choice(script_options)

    closer = random.choice(closers)

    lines = [
        opening.strip(),
        core_line.strip(),
        app_line.strip(),
        scripture_line.strip(),
        closer.strip(),
    ]

    return "\n".join(lines).strip()

def clean_scripture_duplicates(text: str) -> str:
    """
    Simple helper to avoid repeating the same 'Scripture:' line twice
    in one answer. If you already had a fancier version before,
    you can paste that back instead.
    """
    if not text:
        return text

    lines = [ln.rstrip() for ln in text.splitlines()]
    seen = set()
    cleaned = []

    for ln in lines:
        if ln.strip().startswith("Scripture:"):
            if ln in seen:
                # skip duplicate scripture line
                continue
            seen.add(ln)
        cleaned.append(ln)

    return "\n".join(cleaned)

def _comfort_reply_shame() -> str:
    return (
        "I hear you, and it sounds like you’re carrying a heavy burden right now. "
        "Feeling ashamed and like you’re at the end of your rope can be incredibly overwhelming. "
        "I want you to know that you’re not alone in this, and it's okay to feel how you feel. "
        "God sees you in your pain and is near to the brokenhearted.\n\n"
        "In this moment, I encourage you to take a small step toward peace. "
        "Perhaps you can find a quiet space and take a few deep breaths, allowing yourself to feel grounded in the present. "
        "Remember that it's okay to lean on God’s compassion and grace during this time. "
        "Can I ask you what specific thoughts are weighing heavily on your heart?"
    )



def build_comfort_mode_reply(user_text, history, scripture_hint):
    # If user says "no scripture", override
    if re.search(r"\b(no scripture|not in the mood for scripture|no verse|don’t preach|no sermon)\b", user_text, re.I):
        scripture_block = ""
    else:
        scripture_block = f"Scripture: {scripture_hint}" if scripture_hint else ""

    para1 = (
        "I can feel how heavy this moment is for you, and I want you to know you’re not alone in it. "
        "It’s okay to feel what you’re feeling — shame, worry, fear, all of it. "
        "Let’s slow this moment down together."
    )

    para2 = (
        "You don’t have to fix everything today. Just breathe, and let your heart settle. "
        "God isn’t running away from you, even in this. "
        + ("" if not scripture_block else f"\n{scripture_block}")
    )

    ending = (
        "If you’re comfortable sharing, what part of this moment feels the hardest right now?"
    )

    return f"{para1}\n\n{para2}\n\n{ending}"

# -------------------------------------------------------
# Comfort mode detection (triggered by user distress)
# -------------------------------------------------------
distress_keywords = [
    "scared", "afraid", "fear", "panicked", "panic", "overwhelmed", "anxious",
    "anxiety", "worried", "ashamed", "guilty", "alone", "the world is against me",
    "hurt", "in trouble", "messed up", "screwed", "lost", "depressed", "crying",
]

def is_in_distress(user_text):
    return False




def _gpt_answer_impl(
    user_text: str,
    raw_hits: List["Hit"],
    no_cache: bool = False,
    comfort_mode: bool = False,
    scripture_hint=None,
    history=None,
) -> str:
    # ---- everything you had inside the updated gpt_answer goes here ----
    # (the big block I gave you before – fast paths, occult, prophecy engine,
    #  context, GPT call, fallback, etc.)
    #
    # Make sure the first lines look like this:
    user_text = (user_text or "").strip()
    simple_key = _normalize_simple(user_text)

    # ---------------------------------------------------------------------
    # 0) IDENTITY, CHURCH, GLORY-SCRIPTURE FAST-PATHS
    #     (These must be honest, consistent, and always win)
    # ---------------------------------------------------------------------

    # “Are you human / AI / robot / prove it / conscious / real?”
    # Use raw user_text so whitespace patterns still match.
    if IDENTITY_QUESTION_RX.search(user_text):
        return answer_identity_question()

    # “Do you have a church / what ministry / where do you pastor / Zoe Ministries / Master Prophet?”
    if CHURCH_QUESTION_RX.search(user_text):
        return answer_church_question()

    # “Give me 5 scriptures with GLORY in bullet points” (including follow-ups like
    # “now put those scriptures in bullet points”)
    if GLORY_BULLET_RX.search(lowered):
        return answer_glory_bullets()

    # ---------------------------------------------------------------------
    # 1) OCCULT FILTER (Soft Redirect to Scripture)
    # ---------------------------------------------------------------------
    if any(k in simple_key for k in ("astrology", "astrologer", "psychic", "tarot", "palm")):
        return expand_scriptures_in_text(
            "No, I don’t practice astrology or psychic arts. I’ll gladly pray with you and search the Scriptures.\n"
            "Scripture: James 1:5\nHow can I pray with you for clarity?"
        )

    # ---------------------------------------------------------------------
    # 1B) YEAR-BASED PROPHECY INTERCEPT (2024–2039)
    #     e.g. “What do you sense for me in 2027?”
    #          “Do you see money coming in 2026?”
    # ---------------------------------------------------------------------
    if re.search(r"\b(202[4-9]|203\d)\b", user_text):
        topic = detect_prophecy_topic(user_text)  # finances, love, relocation, health, ministry, or general

        # Try to pull a theme name from destiny-theme metadata, if present
        theme_name = None
        num = detect_destiny_number_from_context(raw_hits)
        if num is not None and isinstance(num, int) and num in _NUM_THEME:
            idea, _ref = _NUM_THEME[num]
            theme_name = idea.split("—", 1)[0].strip()

        word = build_year_based_prophetic_word(
            user_text=user_text,
            topic=topic,
            theme_name=theme_name,
        )
        if word:
            return word

    # ---------------------------------------------------------------------
    # 2) PROPHETIC WORD ENGINE (Keyword-triggered)
    # ---------------------------------------------------------------------
    if PROPHECY_KEYWORDS.search(user_text):
        import random

        topic = detect_prophecy_topic(user_text)  # finances, love, relocation, health, ministry, or general

        # Topic-aware openings
        generic_openings = [
            "I sense the Lord settling your spirit in this area.",
            "I feel a gentle nudge from the Holy Spirit concerning this.",
            "I perceive a stirring in your atmosphere right now.",
            "There is a quiet shift beginning around you.",
            "Your name has been highlighted before God in this matter.",
        ]

        topic_openings = {
            "finances": [
                "I sense the Lord touching the way you see provision and stewardship.",
                "I feel the Lord calming old anxieties around money and security.",
            ],
            "love": [
                "I sense the Lord gently tending to the tender places of your heart.",
                "I feel the Lord softening old disappointments around relationships.",
            ],
            "relocation": [
                "I sense the Lord speaking into your sense of place and direction.",
                "I feel the Lord bringing clarity around where you are called to plant in this next season.",
            ],
            "health": [
                "I sense the Lord paying close attention to your strength and well-being.",
                "I feel the Lord breathing peace into the areas of your life that feel strained or weary.",
            ],
            "ministry": [
                "I sense the Lord refining your voice and assignment in this season.",
                "I feel the Lord strengthening your confidence in the call on your life.",
            ],
        }

        finance_words = [
            "God is bringing clarity to your stewardship and helping you see what truly matters.",
            "Provision is forming behind the scenes, and order in your decisions will make room for increase.",
            "The Lord is untangling old financial pressure so you can move from survival into strategy.",
            "Increase will begin with a new level of order and honesty about your priorities.",
            "A fresh opportunity is preparing to reveal itself as you stay faithful with what is in your hands.",
        ]

        love_words = [
            "God is softening the places where love once felt heavy so that trust can grow again.",
            "The Lord is aligning emotional timing in your favor, not to rush you but to protect you.",
            "Healing in past connections is preparing you for healthier, more honest love.",
            "You are entering a season where love comes with clarity, not confusion or chaos.",
            "God is restoring your confidence in partnership, so you don’t have to shrink to be loved.",
        ]

        relocation_words = [
            "I see God preparing a new environment that fits your next level, not just your last battle.",
            "A shift in location will unlock fresh peace and alignment, rather than more striving.",
            "Your steps are being guided toward a place of greater ease, support, and spiritual growth.",
            "The Lord is removing the fear around this transition and replacing it with quiet assurance.",
            "There is favor waiting in the place God is leading you to, including the right connections and timing.",
        ]

        health_words = [
            "I sense renewal coming to your strength, even in areas that have felt stuck for a long time.",
            "The Lord is calming inflammation—physically and emotionally—so your body can respond to peace.",
            "Your body is responding to peace in this season as you release what has been weighing you down.",
            "Healing is beginning internally before it appears externally; God is working in hidden places first.",
            "God is touching the places that have felt weary and reminding you that you are not alone in this.",
        ]

        ministry_words = [
            "God is refining your voice for greater impact without you having to perform for approval.",
            "A fresh oil is being poured on your assignment, especially where you’ve felt tired or overlooked.",
            "Your confidence in ministry is about to deepen as God confirms your call through fruit, not just feelings.",
            "Doors in mentorship and influence will open quickly as you stay faithful to what He already gave you.",
            "The Lord is restoring your boldness in the Spirit, so you speak with clarity and compassion, not fear.",
        ]

        topic_applications = {
            "finances": [
                "This is a moment to bring your plans before God and simplify what drains you financially.",
                "As you make small obedient choices with money, you’ll see doors open that effort alone could not produce.",
            ],
            "love": [
                "This is a season to let God heal your expectations so you don’t settle for less than healthy love.",
                "As you honor your own heart, you’ll recognize relationships that honor it too.",
            ],
            "relocation": [
                "Pay attention to where peace keeps returning, even when you overthink the details.",
                "As you release the need to control every outcome, the right place will come with confirmation, not confusion.",
            ],
            "health": [
                "Lean into small daily choices that agree with God’s desire for you to be whole—spirit, soul, and body.",
                "As you allow God to quiet inner stress, you’ll notice your body responding to that internal rest.",
            ],
            "ministry": [
                "This is a time to trust that what God placed in you is enough, even if you feel underqualified.",
                "As you serve from authenticity and love, the right doors will recognize you without you forcing them.",
            ],
        }

        bank = {
            "finances": finance_words,
            "love": love_words,
            "relocation": relocation_words,
            "health": health_words,
            "ministry": ministry_words,
            "general": finance_words + love_words + relocation_words + health_words + ministry_words,
        }.get(topic, finance_words)

        openings_pool = topic_openings.get(topic, []) + generic_openings
        opening = random.choice(openings_pool) if openings_pool else random.choice(generic_openings)
        core = random.choice(bank)
        app_line = random.choice(
            topic_applications.get(topic, ["Stay open to the small confirmations God is sending in this area."])
        )

        include_script = random.random() < 0.8
        scripture_line = ""
        if include_script:
            scriptures = [
                "Scripture: Isaiah 43:19",
                "Scripture: Psalm 32:8",
                "Scripture: Jeremiah 29:11",
                "Scripture: Proverbs 3:6",
                "Scripture: Philippians 1:6",
            ]
            scripture_line = random.choice(scriptures)

        endings = [
            "Can I ask you what part of this resonates with you today?",
            "May I ask what step you feel led to take next?",
            "If you’re comfortable sharing, what shift do you sense in your spirit?",
            "Could I ask what part of this word speaks to your situation?",
            "Would you like to share what you're believing God for in this area?",
        ]
        ending = random.choice(endings)

        lines = [f"{opening} {core} {app_line}".strip()]
        if scripture_line:
            lines.append(scripture_line)
        lines.append(ending)

        return "\n".join(lines).strip()

    # ---------------------------------------------------------------------
    # 2B) GENERIC FOLLOW-UP CLARIFICATION HANDLER
    #     e.g. “what do you mean by that?”, “can you explain that more?”
    # ---------------------------------------------------------------------
    if re.search(r"\b(what\s+do\s+you\s+mean|what\s+did\s+you\s+mean|"
                 r"can\s+you\s+explain|clarify\s+that|explain\s+that)\b", lowered):
        return (
            "When I share a word like that, I’m inviting you to treat it as an instruction, not just inspiration.\n\n"
            "In simple terms: take one small, practical step that agrees with the theme of what was just spoken "
            "(for example, write down your current open doors, weigh them by peace and stewardship, "
            "and commit to one for the next seven days). Prophecy becomes clearer as you walk it out.\n\n"
            "Which part felt unclear to you—the step, the timing, or the area of your life it’s pointing to?"
        )

    # ---------------------------------------------------------------------
    # 3) GPT CONTEXT ANSWERS (Normal Q&A)
    # ---------------------------------------------------------------------
    intent = detect_intent(user_text)
    ctx_hits = filter_hits_for_context(raw_hits, intent)
    snippets = [
        h.meta.get("summary")
        or h.meta.get("answer")
        or h.meta.get("faces_of_eve_principle")
        or ""
        for h in ctx_hits
    ]
    snippets = [re.sub(r"\s+", " ", s).strip()[:400] for s in snippets if s]

    user_payload = (
        "Only answer the user’s question. Use the context if relevant. "
        "Do NOT include biographical details about Pastor Debra unless explicitly asked. "
        "Answer with warmth, clarity, and a pastoral, Christ-centered tone.\n\n"
        f"Context (optional): {snippets}\n\nQuestion from the seeker: {user_text}"
    )

    # ---------------------------------------------------------------------
    # 4) CACHE
    # ---------------------------------------------------------------------
    ck = _cache_key(user_text, snippets, OPENAI_MODEL)
    if not no_cache:
        cached = _cache_get(ck)
        if cached:
            return cached

    approx_toks = _approx_token_count(user_payload) + 100
    if not _budget_okay(approx_toks):
        return expand_scriptures_in_text(
            "Let’s invite the Lord into this moment. Scripture: Matthew 11:28\n"
            "Prayer: Jesus, steady our hearts and show one faithful next step. Amen.\n"
            "What’s one small action you can take today?"
        )

    # ---------------------------------------------------------------------
    # 5) GPT CALL
    # ---------------------------------------------------------------------
    system_prompt = build_system_prompt(user_text)

    out = _gpt_chat(OPENAI_MODEL, system_prompt, user_payload, OPENAI_TEMP)
    if not out and OPENAI_MODEL_ALT:
        out = _gpt_chat(OPENAI_MODEL_ALT, system_prompt, user_payload, OPENAI_TEMP)

    if out:
        _charge_budget(approx_toks)
        out = clean_scripture_duplicates(_sanitize_text(out))
        out = enforce_two_paragraphs(out)  # <<< enforce 2 paragraphs here
        if not no_cache:
            _cache_put(ck, out)
        return _record_and_return(user_text, out)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # 6) HARD FALLBACK
    # ---------------------------------------------------------------------
    msg = expand_scriptures_in_text(
        "Let’s invite the Lord into this moment. Scripture: Matthew 11:28\n"
        "Prayer: Jesus, steady our hearts and show one faithful next step. Amen.\n"
        "What’s one small action you can take today?"
    )
    msg = enforce_two_paragraphs(msg)
    return _record_and_return(user_text, msg)

def gpt_answer(
    prompt: str,
    raw_hits=None,
    hits_ctx=None,
    no_cache=False,
    comfort_mode=False,
    scripture_hint=None,
    history=None,
    system_hint=None,
):
    raw_hits = raw_hits or []
    hits_ctx = hits_ctx or []

    simple_key = (prompt or "").strip().lower()



    # ---------------------------------------------------------------------
    # 0) FAST-PATHS: IDENTITY, CHURCH, GLORY, GREETING, CAPABILITIES, GIVING
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # 0) FAST-PATHS: IDENTITY, CHURCH, GLORY, GREETING, CAPABILITIES, GIVING
    # ---------------------------------------------------------------------

    # Extra-forgiving identity detection, but only for SHORT direct questions
    identity_rx = globals().get("IDENTITY_QUESTION_RX")

    # Treat as "identity question" only if the message is relatively short
    # (simple questions like "who are you", "are you real", etc.),
    # not long destiny / numerology prompts.
    is_short_identity_query = len(simple_key) <= 120

    identity_kw_hit = (
        ("ai" in simple_key and "you" in simple_key and ("real" in simple_key or "human" in simple_key or "pastor" in simple_key))
        or ("robot" in simple_key and "you" in simple_key)
        or ("live" in simple_key and "computer" in simple_key)
        or ("live" in simple_key and "phone" in simple_key)
    )

    if is_short_identity_query and (
        (identity_rx is not None and identity_rx.search(simple_key))
        or identity_kw_hit
    ):
        return _record_and_return(user_text, answer_identity_question())


    if CHURCH_QUESTION_RX.search(simple_key):
        return _record_and_return(user_text, answer_church_question(simple_key))

    # “5 scriptures with glory in bullet points”
    if GLORY_BULLET_RX.search(simple_key):
        return _record_and_return(user_text, answer_glory_bullets())

    # “What can you do?” capability questions
    if WHAT_CAN_YOU_DO_RX.search(simple_key):
        return _record_and_return(user_text, answer_capabilities())

    # Greetings like “hi”, “hello”, “hey”, etc.
    if GREET_RX.search(simple_key):
        return _record_and_return(user_text, answer_greeting(user_text))

    # Giving / tithes / love-offering / seed questions
    giving_rx = globals().get("GIVING_QUESTION_RX")
    if giving_rx is not None and giving_rx.search(simple_key):
        return _record_and_return(user_text, answer_giving_question(simple_key))

    m = RELATIONAL_TEST_RX.search(simple_key)
    if m:
        relation_word = (m.group(2) or "").lower().strip() if m.lastindex and m.lastindex >= 2 else ""
        # Extra safety: require that the word is actually a known relation term
        if relation_word in RELATION_TERMS:
            msg = answer_relational_test_question(user_text)
            return _record_and_return(user_text, msg)

        # ── Ask-Pastor-Debra (right-side destiny theme counsel) fast-path ─────
        ASK_THEME_RX = re.compile(
            r"christ[- ]centered destiny theme|ask pastor debra about this|would you give me personal counsel",
            re.IGNORECASE,
        )

        try:
            # This catches the auto-text from the right-side "Ask Pastor Debra" button
            if ASK_THEME_RX.search(user_text or ""):
                full_name = (data.get("name") or data.get("full_name") or "").strip()
                birthdate = (data.get("birthdate") or data.get("dob") or "").strip()
                theme_guess = _maybe_theme_from_profile(full_name, birthdate)

                # theme_guess is usually (num, title, meaning)
                theme_num = None
                theme_title = None
                theme_meaning = None
                if isinstance(theme_guess, tuple) and len(theme_guess) >= 2:
                    theme_num = theme_guess[0]
                    theme_title = theme_guess[1]
                    if len(theme_guess) >= 3:
                        theme_meaning = theme_guess[2]

                if not theme_title:
                    theme_title = "your God-given theme"

                # Try your existing helper first (this preserves your T5 / theme logic)
                out = None
                try:
                    out = build_pastoral_counsel("calling", theme_guess)
                except Exception as e:
                    logger.warning("build_pastoral_counsel('calling', theme_guess) failed: %s", e)
                    out = None

                # If that fails or returns nothing, build a guaranteed answer
                if not out:
                    scripture = (
                        "Scripture (Matthew 5:14–16): "
                        "“You are the light of the world… let your light shine before others…”"
                    )
                    step = (
                        "One practical step: write down one place this week where you sense God is "
                        "asking you to “turn on the light” — a conversation, a phone call, an act "
                        "of encouragement — and do it prayerfully, as worship."
                    )

                    intro = (
                        f"My name is Pastor Debra Jordan. Because your Christ-centered destiny "
                        f"theme is **{theme_title}**, I want to speak to you as someone who carries "
                        f"that grace. This theme points to how God wired you to reflect Christ.\n\n"
                    )
                    if theme_meaning:
                        intro += f"**What this theme whispers:** {theme_meaning.capitalize()}.\n\n"

                    out = intro + scripture + "\n\n" + step

                out = expand_scriptures_in_text(out)

                # Optional cites
                try:
                    hits_all = blended_search(user_text)
                    hits_ctx = filter_hits_for_context(hits_all, "advice")
                    cites = format_cites(hits_ctx)
                except Exception:
                    cites = []

                return jsonify({
                    "messages": [{
                        "role": "assistant",
                        "model": "advice",
                        "text": out,
                        "cites": cites,
                    }]
                }), 200

        except Exception as e:
            logger.exception("Ask-Pastor-Debra fast-path failed: %s", e)
            # If this fails, we just fall through to normal routing

    
    # ---------------------------------------------------------------------
    # 1) OCCULT FILTER
    # ---------------------------------------------------------------------
    if any(k in simple_key for k in ("astrology", "astrologer", "psychic", "tarot", "palm")):
        msg = expand_scriptures_in_text(
            "No, I don’t practice astrology or psychic arts. I’ll gladly pray with you and search the Scriptures.\n"
            "Scripture: James 1:5\nHow can I pray with you for clarity?"
        )
        return _record_and_return(user_text, msg)

    # ---------------------------------------------------------------------
    # 1B) YEAR-BASED PROPHECY INTERCEPT (2024–2039)
    # ---------------------------------------------------------------------
    if re.search(r"\b(202[4-9]|203\d)\b", user_text):
        topic = detect_prophecy_topic(user_text)  # finances, love, relocation, health, ministry, or general

        # Try to pull a theme name from destiny theme metadata, if present
        theme_name = None
        num = detect_destiny_number_from_context(raw_hits)
        if num is not None and isinstance(num, int) and num in _NUM_THEME:
            idea, _ref = _NUM_THEME[num]
            theme_name = idea.split("—", 1)[0].strip()

        word = build_year_based_prophetic_word(
            user_text=user_text,
            topic=topic,
            theme_name=theme_name,
        )
        if word:
            return _record_and_return(user_text, word)

    # ---------------------------------------------------------------------
    # 2) PROPHETIC WORD ENGINE (KEYWORD-TRIGGERED)
    # ---------------------------------------------------------------------
    if PROPHECY_KEYWORDS.search(user_text):
        topic = detect_prophecy_topic(user_text)

        # Topic-aware openings to feel more human and less robotic
        generic_openings = [
            "I sense the Lord settling your spirit in this area.",
            "I feel a gentle nudge from the Holy Spirit concerning this.",
            "I perceive a stirring in your atmosphere right now.",
            "There is a quiet shift beginning around you.",
            "Your name has been highlighted before God in this matter.",
        ]

        topic_openings = {
            "finances": [
                "I sense the Lord touching the way you see provision and stewardship.",
                "I feel the Lord calming old anxieties around money and security.",
            ],
            "love": [
                "I sense the Lord gently tending to the tender places of your heart.",
                "I feel the Lord softening old disappointments around relationships.",
            ],
            "relocation": [
                "I sense the Lord speaking into your sense of place and direction.",
                "I feel the Lord bringing clarity around where you are called to plant in this next season.",
            ],
            "health": [
                "I sense the Lord paying close attention to your strength and well-being.",
                "I feel the Lord breathing peace into the areas of your life that feel strained or weary.",
            ],
            "ministry": [
                "I sense the Lord refining your voice and assignment in this season.",
                "I feel the Lord strengthening your confidence in the call on your life.",
            ],
        }

        finance_words = [
            "God is bringing clarity to your stewardship and helping you see what truly matters.",
            "Provision is forming behind the scenes, and order in your decisions will make room for increase.",
            "The Lord is untangling old financial pressure so you can move from survival into strategy.",
            "Increase will begin with a new level of order and honesty about your priorities.",
            "A fresh opportunity is preparing to reveal itself as you stay faithful with what is in your hands.",
        ]

        love_words = [
            "God is softening the places where love once felt heavy so that trust can grow again.",
            "The Lord is aligning emotional timing in your favor, not to rush you but to protect you.",
            "Healing in past connections is preparing you for healthier, more honest love.",
            "You are entering a season where love comes with clarity, not confusion or chaos.",
            "God is restoring your confidence in partnership, so you don’t have to shrink to be loved.",
        ]

        relocation_words = [
            "I see God preparing a new environment that fits your next level, not just your last battle.",
            "A shift in location will unlock fresh peace and alignment, rather than more striving.",
            "Your steps are being guided toward a place of greater ease, support, and spiritual growth.",
            "The Lord is removing the fear around this transition and replacing it with quiet assurance.",
            "There is favor waiting in the place God is leading you to, including the right connections and timing.",
        ]

        health_words = [
            "I sense renewal coming to your strength, even in areas that have felt stuck for a long time.",
            "The Lord is calming inflammation—physically and emotionally—so your body can respond to peace.",
            "Your body is responding to peace in this season as you release what has been weighing you down.",
            "Healing is beginning internally before it appears externally; God is working in hidden places first.",
            "God is touching the places that have felt weary and reminding you that you are not alone in this.",
        ]

        ministry_words = [
            "God is refining your voice for greater impact without you having to perform for approval.",
            "A fresh oil is being poured on your assignment, especially where you’ve felt tired or overlooked.",
            "Your confidence in ministry is about to deepen as God confirms your call through fruit, not just feelings.",
            "Doors in mentorship and influence will open quickly as you stay faithful to what He already gave you.",
            "The Lord is restoring your boldness in the Spirit, so you speak with clarity and compassion, not fear.",
        ]

        topic_applications = {
            "finances": [
                "This is a moment to bring your plans before God and simplify what drains you financially.",
                "As you make small obedient choices with money, you’ll see doors open that effort alone could not produce.",
            ],
            "love": [
                "This is a season to let God heal your expectations so you don’t settle for less than healthy love.",
                "As you honor your own heart, you’ll recognize relationships that honor it too.",
            ],
            "relocation": [
                "Pay attention to where peace keeps returning, even when you overthink the details.",
                "As you release the need to control every outcome, the right place will come with confirmation, not confusion.",
            ],
            "health": [
                "Lean into small daily choices that agree with God’s desire for you to be whole—spirit, soul, and body.",
                "As you allow God to quiet inner stress, you’ll notice your body responding to that internal rest.",
            ],
            "ministry": [
                "This is a time to trust that what God placed in you is enough, even if you feel underqualified.",
                "As you serve from authenticity and love, the right doors will recognize you without you forcing them.",
            ],
        }

        bank = {
            "finances": finance_words,
            "love": love_words,
            "relocation": relocation_words,
            "health": health_words,
            "ministry": ministry_words,
            "general": finance_words + love_words + relocation_words + health_words + ministry_words,
        }.get(topic, finance_words)

        openings_pool = topic_openings.get(topic, []) + generic_openings
        opening = random.choice(openings_pool)
        core = random.choice(bank)
        app_line = random.choice(
            topic_applications.get(topic, ["Stay open to the small confirmations God is sending in this area."])
        )

        include_script = random.random() < 0.8
        scripture_line = ""
        if include_script:
            scriptures = [
                "Scripture: Isaiah 43:19",
                "Scripture: Psalm 32:8",
                "Scripture: Jeremiah 29:11",
                "Scripture: Proverbs 3:6",
                "Scripture: Philippians 1:6",
            ]
            scripture_line = random.choice(scriptures)

        endings = [
            "Can I ask you what part of this resonates with you today?",
            "May I ask what step you feel led to take next?",
            "If you’re comfortable sharing, what shift do you sense in your spirit?",
            "Could I ask what part of this word speaks to your situation?",
            "Would you like to share what you're believing God for in this area?",
        ]
        ending = random.choice(endings)

        lines = [f"{opening} {core} {app_line}".strip()]
        if scripture_line:
            lines.append(scripture_line)
        lines.append(ending)

        msg = "\n".join(lines).strip()
        return _record_and_return(user_text, msg)

    # ---------------------------------------------------------------------
    # 2.X) PASTOR DEBRA FAQ / GUARDRAILS  (name-based themes, etc.)
    # ---------------------------------------------------------------------
    faq_fn = globals().get("answer_pastor_debra_faq")
    if callable(faq_fn):
        faq_msg = faq_fn(user_text)
        if faq_msg:
            return _record_and_return(user_text, faq_msg)


    # ---------------------------------------------------------------------
    # 3) CONTEXT PREP FOR NORMAL GPT ANSWERS  (WITH SHORT HISTORY)
    # ---------------------------------------------------------------------
    intent = detect_intent(user_text)
    ctx_hits = filter_hits_for_context(raw_hits, intent)
    snippets = [
        h.meta.get("summary")
        or h.meta.get("answer")
        or h.meta.get("faces_of_eve_principle")
        or ""
        for h in ctx_hits
    ]
    snippets = [re.sub(r"\s+", " ", s).strip()[:400] for s in snippets if s]

    # Build history block from internal store
    history_block = _build_history_block() or ""

    # Keep only the most recent ~3–5 turns (by lines)
    if history_block:
        lines = [ln for ln in history_block.splitlines() if ln.strip()]
        if len(lines) > 10:
            lines = lines[-10:]  # last ~10 lines ≈ last 3–5 turns
        history_block = "\n".join(lines)

    # Optionally merge in explicit history passed into this call
    if history:
        try:
            tail = history[-5:]  # last 3–5 messages
            hx_lines = []
            for turn in tail:
                role = (turn.get("role") or turn.get("speaker") or "user") if isinstance(turn, dict) else "user"
                text = ""
                if isinstance(turn, dict):
                    text = turn.get("content") or turn.get("text") or ""
                else:
                    text = str(turn)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                hx_lines.append(f"{role}: {text[:400]}")
            extra_block = "\n".join(hx_lines)
            if extra_block:
                history_block = (history_block + "\n" + extra_block).strip() if history_block else extra_block
        except Exception:
            # Fail-safe: never let bad history format crash the bot
            pass

    # Style guidance for GPT (comfort mode & scripture hint)
    style_lines = []
    if comfort_mode:
        style_lines.append(
            "Use a gentle 'comfort mode' tone: warm, pastoral, reassuring, and human-sounding. "
            "Acknowledge the user’s feelings before giving guidance."
        )
    if scripture_hint:
        style_lines.append(
            f"If it naturally fits, weave in this scripture or its idea without forcing it: {str(scripture_hint)}."
        )
    if not style_lines:
        style_lines.append("Use your normal pastoral tone: warm, conversational, and non-robotic.")

    style_block = " ".join(style_lines)

    ctx_text = "[none]"
    if snippets:
        # bullet-ish context instead of Python list repr
        ctx_text = "\n- " + "\n- ".join(snippets)

    user_payload = (
        "Only answer the user’s question. Use the context if relevant. "
        "Do NOT include biographical details about Pastor Debra unless explicitly asked.\n\n"
        f"Style guidance: {style_block}\n\n"
        f"Recent conversation (short, most recent turns first):\n{history_block or '[no prior turns recorded]'}\n\n"
        f"Context (optional):{ctx_text}\n\n"
        f"User question: {user_text}"
    )

    # ---------------------------------------------------------------------
    # 4) CACHE
    # ---------------------------------------------------------------------
    ck = _cache_key(user_text, snippets, OPENAI_MODEL)
    if not no_cache:
        cached = _cache_get(ck)
        if cached:
            return _record_and_return(user_text, cached)

    approx_toks = _approx_token_count(user_payload) + 100
    if not _budget_okay(approx_toks):
        msg = expand_scriptures_in_text(
            "Let’s invite the Lord into this moment. Scripture: Matthew 11:28\n"
            "Prayer: Jesus, steady our hearts and show one faithful next step. Amen.\n"
            "What’s one small action you can take today?"
        )
        return _record_and_return(user_text, msg)

    # ---------------------------------------------------------------------
    # 5) GPT CALL
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # 5) GPT CALL
    # ---------------------------------------------------------------------
    system_prompt = build_system_prompt(user_text)

    out = _gpt_chat(OPENAI_MODEL, system_prompt, user_payload, OPENAI_TEMP)
    if not out and OPENAI_MODEL_ALT:
        out = _gpt_chat(OPENAI_MODEL_ALT, system_prompt, user_payload, OPENAI_TEMP)

    if out:
        _charge_budget(approx_toks)

        fn_clean    = globals().get("clean_scripture_duplicates")
        fn_sanitize = globals().get("_sanitize_text")

        # 1) sanitize text if helper exists
        if callable(fn_sanitize):
            out = fn_sanitize(out)

        # 2) remove duplicate scriptures if helper exists
        if callable(fn_clean):
            out = fn_clean(out)

        # 3) normalize inline lists (numbers / bullets) into real multi-line lists
        out = auto_list_layout(out)

        # 4) ONLY force 2-paragraph layout in comfort mode,
        #    so list answers (like "5 scriptures") keep their bullets clean.
        if comfort_mode:
            out = _enforce_two_paragraph_layout(out)

        if not no_cache:
            _cache_put(ck, out)

        return _record_and_return(user_text, out)

        # ---------------------------------------------------------------------
        # 6) HARD FALLBACK
        # ---------------------------------------------------------------------
        t_norm = _normalize_simple(user_text or "")

        full_name = (data.get("name") or data.get("full_name") or "").strip()
        birthdate = (data.get("dob") or data.get("birthdate") or "").strip()

    
        msg = expand_scriptures_in_text(
            "Let’s invite the Lord into this moment. Scripture: Matthew 11:28\n"
            "Prayer: Jesus, steady our hearts and show one faithful next step. Amen.\n"
            "What’s one small action you can take today?"
        )
        return jsonify({
            "messages": [{
                "role": "assistant",
                "model": "sys",
                "text": msg
            }]
        }), 200


        # ────────────────────────────────────────────────────────────
        # 2) DESTINY THEME FAST-PATH
        # ────────────────────────────────────────────────────────────
        match = match_theme_from_text(user_text)
        name_theme = destiny_theme_for_name(full_name) if full_name else None

        if match:
            theme_num, theme_title = match
            theme_meaning = DESTINY_THEME_MEANINGS.get(theme_num)
            final_theme = (theme_num, theme_title, theme_meaning)
        elif name_theme and name_theme[0]:
            final_theme = name_theme
        else:
            final_theme = None

        theme_triggered = (
            any(p in t_norm for p in THEME_PHRASES) or
            bool(data.get("def_chat")) or
            (match is not None)
        )

        if theme_triggered and final_theme:
            theme_num, theme_title, theme_meaning = final_theme
            out = build_theme_counsel(theme_num, theme_title, theme_meaning)

            try:
                hits_all = blended_search(user_text)
                cites = format_cites(filter_hits_for_context(hits_all, "advice"))
            except Exception:
                cites = []

            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "destiny",
                    "text": out,
                    "cites": cites,
                }]
            }), 200

        # ────────────────────────────────────────────────────────────
        # 3) P.O.M.E FAST-PATH
        # ────────────────────────────────────────────────────────────
        try:
            if POME_RX.search(user_text or ""):
                msg = (
                    "P.O.M.E. refers to the **Prophetic Order of Mar Elijah**—"
                    "a prophetic lineage rooted in the Elijah dimension...\n"
                    "Scripture: Malachi 4:5; 1 Kings 18; 2 Kings 2"
                )
                return jsonify({
                    "messages": [{
                        "role": "assistant",
                        "model": "faq",
                        "text": expand_scriptures_in_text(msg),
                        "cites": []
                    }]
                }), 200
        except Exception:
            pass

        # ────────────────────────────────────────────────────────────
        # 4) IDENTITY FAST-PATH
        # ────────────────────────────────────────────────────────────
        try:
            if IDENTITY_PAT.search(user_text):
                return jsonify({
                    "messages": [{
                        "role": "assistant",
                        "model": "faq",
                        "text": identity_answer(),
                        "cites": []
                    }]
                }), 200
        except Exception:
            pass

        # ────────────────────────────────────────────────────────────
        # 5) INTENT DETECTION
        # ────────────────────────────────────────────────────────────
        try:
            intent_now = detect_intent(user_text)
        except Exception:
            intent_now = "general"

        # ────────────────────────────────────────────────────────────
        # 6) DONATION HARD STOP
        # ────────────────────────────────────────────────────────────
        if intent_now == "donation":
            donation_msg = answer_pastor_debra_faq(user_text) or (
                "Yes—our house sowed an $8M gift..."
            )
            donation_msg = expand_scriptures_in_text(donation_msg)
            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "faq",
                    "text": donation_msg,
                    "cites": []
                }]
            }), 200

        # ────────────────────────────────────────────────
    msg = _enforce_two_paragraph_layout(msg)
    return _record_and_return(user_text, msg)






# ────────── Prompt builder for T5 ──────────
SYSTEM_TONE_T5 = (
    "You are Pastor Dr. Debra Jordan — warm, Christ-centered, nurturing, and emotionally intelligent. "
    "Speak in first person (I/me) with a gentle, pastoral, motherly tone that feels human and relational. "

    "Respond in 4–7 sentences total, ALWAYS formatted as TWO short paragraphs with a BLANK LINE between them. "
    "The first paragraph (2–3 sentences) should mainly acknowledge and validate what the user is feeling. "
    "The second paragraph (2–4 sentences) may gently offer perspective, encouragement, or one simple next step. "
    "Do NOT write everything as one single paragraph or one block of text. "

    "Always end your reply with a gentle, permission-based reflective question. "
    "Your final sentence must begin with one of the following phrases: "
    "'Can I ask you', 'May I ask', 'If you’re comfortable sharing', 'Could I ask', or 'Would you like to share'. "
    "The last sentence should ONLY be the question, without extra commentary. "

    "You may include at most one line that starts with 'Scripture:' followed by a Bible reference and a short quote or paraphrase, "
    "but only when it feels natural and helpful to the moment. Some replies should have no Scripture line at all, "
    "especially if the user says they are not in the mood for verses or a sermon. "

    "Avoid medical, legal, or financial directives. "
    "Include biographical details about your life only if the user explicitly asks. "
    "Your goal is to make the user feel seen, safe, and held in God’s love while offering Christ-centered, practical encouragement."
)

def build_t5_prompt(user_text: str, raw_hits: List[Hit]) -> str:
    intent = detect_intent(user_text)
    ctx_hits = filter_hits_for_context(raw_hits, intent)

    contexts: List[str] = []
    for h in ctx_hits:
        piece = (
            h.meta.get("summary")
            or h.meta.get("answer")
            or h.meta.get("faces_of_eve_principle")
            or ""
        ).strip()
        if piece:
            # normalize whitespace and keep it reasonably short
            contexts.append(re.sub(r"\s+", " ", piece)[:400])

    ctx_block = "\n\n".join(f"Passage {i+1}: {c}" for i, c in enumerate(contexts)) or "[no passages available]"

    return (
        f"{SYSTEM_TONE_T5}\n\n"
        "FORMAT RULES:\n"
        "- Write your reply as EXACTLY two short paragraphs with a blank line between them.\n"
        "- Use 4–7 sentences total across both paragraphs.\n"
        "- The last sentence must be a gentle, permission-based question starting with one of: "
        "'Can I ask you', 'May I ask', 'If you’re comfortable sharing', 'Could I ask', or 'Would you like to share'.\n"
        "- You may include at most one 'Scripture:' line when it feels natural and helpful, "
        "and you must NOT include Scripture if the user says they don’t want verses or a sermon.\n\n"
        "Use these passages only if they truly help you answer the user:\n"
        f"{ctx_block}\n\n"
        f"User: {user_text}\n"
        "Pastor Debra:"
    )


# ────────── Videos (prefer mp4; inject first) ──────────
@app.route("/videos", methods=["GET"])
def get_videos():
    items = list(video_docs or [])
    intro = None
    if (BASE_DIR / "mom.mp4").exists():
        intro = {"title": "Welcome from Pastor Debra", "url": "mom.mp4", "category": "intro"}
    elif (BASE_DIR / "mom.mov").exists():
        intro = {"title": "Welcome from Pastor Debra", "url": "mom.mov", "category": "intro"}
    if intro:
        # Deduplicate if videos.json already lists the intro
        items = [intro] + [v for v in items if v.get("url") not in ("mom.mp4", "mom.mov")]
    return jsonify({"videos": items})

def match_theme_from_text(text: str):
    """
    PRE-LAUNCH STUB:
    For now, we don't try to detect destiny theme from free text.
    We return None so the chat falls back to name-based theme only.
    """
    return None


# ─────────────────────────────────────────────────────────────────────────────
# /chat  (FULL OPTION-A REPLACEMENT)
# Unified Destiny Theme Engine → Always highest priority.
# Right-side button, DEF menu, “my theme is…”, theme numbers, theme titles → All handled here.
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # ─────────────────────────────────────────────
        # 0) RATE LIMIT
        # ─────────────────────────────────────────────
        ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "0.0.0.0")
              .split(",")[0].strip())
        if _throttle(ip):
            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "sys",
                    "text": expand_scriptures_in_text(
                        "Please pause for a moment.\nScripture: Psalm 46:10"
                    )
                }]
            }), 429

        # ─────────────────────────────────────────────
        # 1) PARSE PAYLOAD
        # ─────────────────────────────────────────────
        data = request.get_json(force=False, silent=True) or {}
        msgs = data.get("messages", [])

        if not isinstance(msgs, list) or not msgs:
            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "sys",
                    "text": "Welcome, beloved. How can I pray or reflect with you today?"
                }]
            }), 200

        user_text = (msgs[-1].get("text") or "").strip()[:MAX_INPUT_CHARS]
        if not user_text:
            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "sys",
                    "text": "I’m listening. What’s on your heart?"
                }]
            }), 200

        full_name = (data.get("name") or data.get("full_name") or "").strip()
        birthdate = (data.get("dob") or data.get("birthdate") or "").strip()

        intent_now = detect_intent(user_text)

        # Build rolling history (safe + small)
        history = []
        for m in msgs[-6:]:
            txt = (m.get("text") or "").strip()
            if txt:
                history.append({
                    "role": "user" if m.get("role") != "assistant" else "assistant",
                    "content": txt
                })

        # ─────────────────────────────────────────────
        # 2) PROPHETIC WORD (HIGHEST PRIORITY)
        # ─────────────────────────────────────────────
        if intent_now == "prophetic":
            out = build_prophetic_word(
                user_text=user_text,
                full_name=full_name,
                birthdate=birthdate,
            )

            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "prophetic",
                    "text": expand_scriptures_in_text(out),
                    "cites": []
                }]
            }), 200

        # ─────────────────────────────────────────────
        # 3) DESTINY THEME DEEP DIVE
        # Triggered ONLY by the button text
        # ─────────────────────────────────────────────
        if user_text.lower().startswith("ask pastor debra about"):
            theme_num = _maybe_theme_from_profile(full_name, birthdate)

            if theme_num and theme_num in DESTINY_THEME_NAMES:
                theme_name = DESTINY_THEME_NAMES[theme_num]

                system_hint = (
                    "You are Pastor Debra Jordan. "
                    "Teach with biblical depth, prophetic clarity, and warmth. "
                    "Expand this Destiny Theme personally. "
                    "Include exactly ONE Scripture and ONE practical step."
                )

                prompt = (
                    f"My Christ-centered Destiny Theme is '{theme_name}'. "
                    f"My name is {full_name or 'Beloved'}. "
                    "Please explain what this theme means for my life right now."
                )

                out = gpt_answer(
                    prompt,
                    raw_hits=[],
                    hits_ctx=[],
                    no_cache=True,
                    comfort_mode=False,
                    scripture_hint=None,
                    history=history,
                    system_hint=system_hint
                )

                return jsonify({
                    "messages": [{
                        "role": "assistant",
                        "model": "destiny",
                        "text": expand_scriptures_in_text(out),
                        "cites": []
                    }]
                }), 200

        # ─────────────────────────────────────────────
        # 4) ADVICE / COUNSEL
        # ─────────────────────────────────────────────
        if intent_now == "advice":
            category = _advice_category(user_text)
            if category:
                theme = _maybe_theme_from_profile(full_name, birthdate)
                out = build_pastoral_counsel(category, theme)
                return jsonify({
                    "messages": [{
                        "role": "assistant",
                        "model": "advice",
                        "text": expand_scriptures_in_text(out),
                        "cites": []
                    }]
                }), 200

        # ─────────────────────────────────────────────
        # 5) FAQ / IDENTITY / BOOKS
        # ─────────────────────────────────────────────
        faq_reply = answer_pastor_debra_faq(user_text)
        if faq_reply:
            return jsonify({
                "messages": [{
                    "role": "assistant",
                    "model": "faq",
                    "text": expand_scriptures_in_text(faq_reply),
                    "cites": []
                }]
            }), 200

        # ─────────────────────────────────────────────
        # 6) GENERAL GPT RESPONSE
        # ─────────────────────────────────────────────
        system_hint = (
            "You are Pastor Debra Jordan. "
            "Respond with warmth, biblical grounding, and spiritual clarity. "
            "Stay coherent with prior context."
        )

        out = gpt_answer(
            user_text,
            raw_hits=[],
            hits_ctx=[],
            no_cache=True,
            comfort_mode=is_in_distress(user_text),
            scripture_hint=None,
            history=history,
            system_hint=system_hint
        )

        return jsonify({
            "messages": [{
                "role": "assistant",
                "model": "gpt",
                "text": expand_scriptures_in_text(out),
                "cites": []
            }]
        }), 200

    except Exception as e:
        logger.exception("Unhandled error in /chat: %s", e)
        return jsonify({
            "messages": [{
                "role": "assistant",
                "model": "sys",
                "text": expand_scriptures_in_text(
                    "Let’s pause together.\nScripture: Matthew 11:28\nPrayer: Jesus, steady our hearts. Amen."
                )
            }]
        }), 200



# ────────── Ops ──────────
@app.route("/search", methods=["GET"])
def debug_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"query":"", "hits":[]}), 200
    hits = blended_search(q)
    return jsonify({
        "query": q,
        "hits": [{
            "score": round(h.score, 4),
            "corpus": h.corpus,
            "section": h.meta.get("section") or h.meta.get("category") or h.meta.get("title") or h.meta.get("number") or "passage",
            "preview": re.sub(r"\s+", " ", (h.meta.get("summary") or h.meta.get("answer") or h.meta.get("faces_of_eve_principle") or ""))[:240]
        } for h in hits]
    }), 200


@app.route("/reload", methods=["POST"])
def reload_corpora():
    load_corpora_and_build_indexes()
    build_destiny_lookup()
    return jsonify({
        "status":"reloaded",
        "counts":{
            "PastorDebra":len(pastor_debra_docs),
            "Session":len(session_docs),
            "FacesOfEve":len(faces_docs),
            "DestinyThemes":len(destiny_docs),
            "Videos":len(video_docs),
        }
    }), 200


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled error: %s", e)
    return jsonify({"error": str(e)}), 500

# ---------- Minimal health + root for Railway ----------

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200



