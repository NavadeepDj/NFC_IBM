import base64
import io
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
ARCFACE_PATH = os.getenv("ARC_FACE_ONNX_PATH", "arcface.onnx")
ENROLLED_PATH = os.getenv("ENROLLED_EMBEDDINGS_PATH", "enrolled_embeddings.json")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.42"))

app = Flask(__name__, static_folder=".")
CORS(app)

# ------------------------------------------------------------
# Load ArcFace ONNX
# ------------------------------------------------------------
ort_session = None
input_name = None
input_layout = "NCHW"  # default; will detect
try:
    if os.path.exists(ARCFACE_PATH):
        ort_session = ort.InferenceSession(
            ARCFACE_PATH, providers=["CPUExecutionProvider"]
        )
        inp = ort_session.get_inputs()[0]
        input_name = inp.name
        shape = inp.shape  # e.g., [None, 3, 112, 112] or [1, 112, 112, 3]
        # Resolve layout
        resolved = [s if isinstance(s, int) else 1 for s in shape]
        if len(resolved) >= 4:
            if resolved[1] == 3:
                input_layout = "NCHW"
            elif resolved[-1] == 3:
                input_layout = "NHWC"
        print(f"✅ Loaded ArcFace ONNX: {ARCFACE_PATH} | layout: {input_layout} | shape: {shape}")
    else:
        print(f"⚠️ ArcFace ONNX not found at {ARCFACE_PATH}")
except Exception as e:
    print(f"❌ Failed to load ArcFace ONNX: {e}")
    ort_session = None

# ------------------------------------------------------------
# Load enrolled embeddings
# Format: { "student_id": [float embedding array] }
# ------------------------------------------------------------
def load_enrolled():
    if not os.path.exists(ENROLLED_PATH):
        print(f"⚠️ Enrolled embeddings file not found: {ENROLLED_PATH}")
        return {}
    try:
        with open(ENROLLED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded enrolled embeddings: {len(data)} students")
        return data
    except Exception as e:
        print(f"❌ Failed to load enrolled embeddings: {e}")
        return {}


def _normalize_vec(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(float).tolist()


enrolled_embeddings = load_enrolled()
# Normalize all loaded embeddings defensively
for sid, vec in list(enrolled_embeddings.items()):
    enrolled_embeddings[sid] = _normalize_vec(vec)


# ------------------------------------------------------------
# Face detector (OpenCV Haar as fallback, no TF/mediapipe deps)
# ------------------------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_detector.empty():
    face_detector = None
    print("⚠️ Haar face detector not available.")
else:
    print("✅ Haar face detector loaded.")


def detect_face_bbox(image_bgr):
    """
    Return (x1, y1, x2, y2) for largest detected face.
    Fallbacks:
      - second pass with more sensitive params
      - center crop if nothing detected
    """
    h, w, _ = image_bgr.shape

    def center_crop_bbox():
        side = int(min(h, w) * 0.6)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        return x1, y1, x1 + side, y1 + side

    if face_detector is None:
        print("⚠️ Face detector unavailable; using center crop fallback")
        return center_crop_bbox()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Pass 1: default params
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        # Pass 2: more sensitive
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(60, 60)
        )

    if len(faces) == 0:
        print("⚠️ No face detected; using center crop fallback")
        return center_crop_bbox()

    # choose largest face
    x, y, w_box, h_box = max(faces, key=lambda r: r[2] * r[3])
    pad_x = int(w_box * 0.2)
    pad_y = int(h_box * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + w_box + pad_x)
    y2 = min(h, y + h_box + pad_y)
    return x1, y1, x2, y2


# ------------------------------------------------------------
# ArcFace inference helpers
# ------------------------------------------------------------
def preprocess_arcface(face_bgr):
    """Return tensor shaped to match model layout (NCHW or NHWC), float32 normalized."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    face = resized.astype(np.float32)
    face = (face - 127.5) / 128.0
    if input_layout == "NHWC":
        face = face[None, ...]  # (1,112,112,3)
    else:  # NCHW
        face = np.transpose(face, (2, 0, 1))[None, ...]  # (1,3,112,112)
    return face.astype(np.float32)


def get_embedding(face_bgr):
    if ort_session is None or input_name is None:
        raise RuntimeError("ArcFace model not loaded")
    tensor = preprocess_arcface(face_bgr)
    outputs = ort_session.run(None, {input_name: tensor})
    emb = outputs[0].reshape(-1)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(float)


def best_match(emb, enrolled, threshold):
    """Return (best_id, best_similarity, is_match) with cosine similarity."""
    best_id = None
    best_sim = -1.0
    for sid, vec in enrolled.items():
        vec = np.array(vec, dtype=np.float32)
        # Normalize stored embedding defensively (in case)
        vnorm = np.linalg.norm(vec)
        if vnorm > 0:
            vec = vec / vnorm
        sim = float(np.dot(emb, vec))
        if sim > best_sim:
            best_sim = sim
            best_id = sid
    is_match = best_sim >= threshold
    return best_id, best_sim, is_match


# ------------------------------------------------------------
# Routes: static pages (reuse existing frontend)
# ------------------------------------------------------------
@app.route("/")
@app.route("/index")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")


@app.route("/facultylogin")
@app.route("/faculty-login")
@app.route("/facultylogin.html")
def faculty_login():
    return send_from_directory(".", "faculty-login.html")


@app.route("/aboutus")
@app.route("/aboutus.html")
def aboutus():
    return send_from_directory(".", "aboutus.html")


@app.route("/faculty-dashboard-fixed.html")
def faculty_dashboard():
    return send_from_directory(".", "faculty-dashboard-fixed.html")


@app.route("/register-face")
@app.route("/register-face.html")
def register_face_page():
    return send_from_directory(".", "register-face.html")


@app.route("/test-arcface")
@app.route("/test-arcface.html")
def test_arcface_page():
    return send_from_directory(".", "test-arcface.html")


@app.route("/<path:filename>")
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory(".", filename)
    return "File not found", 404


# ------------------------------------------------------------
# API: verify via ArcFace embedding + NFC
# ------------------------------------------------------------
@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.json or {}
        image_data = data.get("image", "")
        nfc_id = str(data.get("nfc_id", "")).strip()

        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # strip data URL prefix
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # detect face
        bbox = detect_face_bbox(img_bgr)
        if bbox is None:
            return jsonify(
                {
                    "success": False,
                    "error": "No face detected",
                    "detected_persons": [],
                    "nfc_id": nfc_id,
                }
            )

        x1, y1, x2, y2 = bbox
        face_bgr = img_bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            return jsonify(
                {
                    "success": False,
                    "error": "Face crop failed",
                    "detected_persons": [],
                    "nfc_id": nfc_id,
                }
            )

        # get embedding
        emb = get_embedding(face_bgr)

        # match
        matched_id, sim, _ = best_match(
            emb, enrolled_embeddings, SIMILARITY_THRESHOLD
        )

        similarity_pct = sim * 100.0
        # New logic: no similarity threshold for match decision; just nearest ID == NFC
        THRESHOLD_PCT = 0.0
        is_match = matched_id is not None and matched_id == nfc_id

        print(
            f"[VERIFY] nfc_id={nfc_id} matched_id={matched_id} "
            f"similarity={sim:.4f} ({similarity_pct:.1f}%) "
            f"is_match={is_match} (no similarity threshold enforced)"
        )

        result = {
            "success": True,
            "nfc_id": nfc_id,
            "matched_id": matched_id,
            "similarity": sim,
            "similarity_percent": similarity_pct,
            "threshold": 0.0,
            "threshold_percent": THRESHOLD_PCT,
            "is_match": is_match,
            "detected_persons": [],
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def save_enrolled():
    try:
        with open(ENROLLED_PATH, "w", encoding="utf-8") as f:
            json.dump(enrolled_embeddings, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Failed to save enrolled embeddings: {e}")
        return False


@app.route("/enroll", methods=["POST"])
def enroll():
    """Register a face embedding for a given student_id."""
    try:
        data = request.json or {}
        image_data = data.get("image", "")
        student_id = str(data.get("student_id", "")).strip()

        if not student_id:
            return jsonify({"success": False, "error": "student_id is required"}), 400

        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        bbox = detect_face_bbox(img_bgr)
        if bbox is None:
            return jsonify({"success": False, "error": "No face detected"}), 400

        x1, y1, x2, y2 = bbox
        face_bgr = img_bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            return jsonify({"success": False, "error": "Face crop failed"}), 400

        emb = get_embedding(face_bgr)
        emb = _normalize_vec(emb)
        enrolled_embeddings[student_id] = emb
        if save_enrolled():
            return jsonify(
                {
                    "success": True,
                    "student_id": student_id,
                    "message": "Enrollment saved",
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return jsonify({"success": False, "error": "Failed to save enrollment"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "arcface_loaded": ort_session is not None,
            "enrolled_count": len(enrolled_embeddings),
            "threshold": SIMILARITY_THRESHOLD,
            "arcface_path": ARCFACE_PATH,
        }
    )


if __name__ == "__main__":
    print("🚀 Starting ArcFace server on http://0.0.0.0:5000")
    print("📡 POST /verify with { image: dataURL, nfc_id: '...' }")
    app.run(host="0.0.0.0", port=5000, debug=True)

