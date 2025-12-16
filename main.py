from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import base64
import io

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__, static_folder='.')
CORS(app)  # Allow frontend JS to call our APIs (safe even on same origin)

# -----------------------------
# Model loading (YOLOv8)
# -----------------------------
MODEL_PATH = "best (15).pt"
model = YOLO(MODEL_PATH)

print(f"✅ Loaded face recognition model: {MODEL_PATH}")
print(f"📋 Model trained with classes: {model.names}")


# -----------------------------
# Static page routes
# -----------------------------
@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/aboutus')
@app.route('/aboutus.html')
def aboutus():
    return send_from_directory('.', 'aboutus.html')


@app.route('/facultylogin')
@app.route('/faculty-login')
@app.route('/facultylogin.html')
def faculty_login():
    return send_from_directory('.', 'faculty-login.html')


@app.route('/faculty-dashboard-fixed.html')
def faculty_dashboard():
    return send_from_directory('.', 'faculty-dashboard-fixed.html')


# -----------------------------
# Face verification API (MUST be before catch-all route!)
# -----------------------------
@app.route('/verify', methods=['POST'])
def verify_face():
    """
    Verify face from image against NFC ID using YOLO model.
    Expects JSON: { "image": "base64_image", "nfc_id": "student_id" }
    Returns JSON: { "success": true/false, "detected_persons": [...], "nfc_id": "..." }
    """
    try:
        data = request.json or {}

        # Get base64 image and NFC ID
        image_data = data.get('image', '')
        nfc_id = str(data.get('nfc_id', '')).strip()

        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image data provided",
                "detected_persons": []
            }), 400

        print(f"📥 Received image data length: {len(image_data)} chars")
        print(f"📥 NFC ID received: '{nfc_id}'")

        # Remove data URL prefix if present (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 to image
        img_bytes = base64.b64decode(image_data)
        print(f"📥 Decoded image bytes: {len(img_bytes)} bytes")
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)

        print(f"📸 Image decoded: {img_array.shape} (HxWxC)")

        # Convert RGB to BGR for OpenCV/YOLO
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Try multiple orientations (helps with rotated phone images)
        images_to_try = [
            img_bgr,
            cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_180),
        ]

        best_results = []
        found_match = False

        print(f"\n🔍 Verifying NFC ID: '{nfc_id}'")

        for i, current_img in enumerate(images_to_try):
            print(f"  👉 Trying orientation {i + 1}/4...")
            # Lower confidence threshold to 0.1 (default is 0.25, might be too high)
            results = model(current_img, verbose=False, conf=0.1)
            
            # Debug: count total boxes found
            total_boxes = 0
            for result in results:
                total_boxes += len(result.boxes)
            print(f"     📦 Found {total_boxes} detection(s) in this orientation")

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = str(model.names[class_id]).strip()

                    print(f"     👀 Detected: '{class_name}' (confidence: {confidence:.3f})")

                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": confidence,
                    }
                    best_results.append(detection)

                    # Check for match (case insensitive)
                    if class_name.lower() == nfc_id.lower():
                        found_match = True
                        # Prioritize this result
                        best_results = [detection] + best_results
                        print(f"     ✅ MATCH FOUND! '{class_name}' == '{nfc_id}'")
                        break

            if found_match:
                print("  ✅ Match found in this orientation!")
                break

        # Remove duplicate class names
        unique_results = []
        seen = set()
        for res in best_results:
            if res["class"] not in seen:
                unique_results.append(res)
                seen.add(res["class"])

        print(f"📊 Final Results: {unique_results}")

        return jsonify(
            {
                "success": True,
                "detected_persons": unique_results,
                "nfc_id": nfc_id,
            }
        )

    except Exception as e:
        print(f"❌ Error in /verify: {str(e)}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "detected_persons": [],
                }
            ),
            500,
        )


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for the face recognition API."""
    return jsonify(
        {
            "status": "healthy",
            "model": MODEL_PATH,
            "classes": model.names,
        }
    )


# -----------------------------
# Catch-all route for static files (MUST be last!)
# -----------------------------
@app.route('/<path:filename>')
def serve_file(filename):
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404


if __name__ == '__main__':
    print("🚀 Starting combined app (static + YOLO API) on http://0.0.0.0:5000")
    print("📡 Face verify endpoint: POST /verify")
    print("🏥 Health check: GET /health")
    app.run(debug=True, host='0.0.0.0', port=5000)


