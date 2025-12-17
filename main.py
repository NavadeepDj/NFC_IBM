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
# Model loading (YOLOv8 + OpenCV DNN Face Detector)
# -----------------------------
MODEL_PATH = "best (15).pt"
model = YOLO(MODEL_PATH)

# Initialize OpenCV DNN Face Detector (using pre-trained model)
# Download the model files if not present
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"

# Try to load the DNN face detector, fallback to Haar Cascade if not available
face_detector = None
try:
    # Try DNN first (more accurate)
    if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
        face_detector = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
        print("✅ Loaded OpenCV DNN face detector")
    else:
        # Fallback to Haar Cascade (built-in, less accurate but works)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not face_cascade.empty():
            face_detector = face_cascade
            print("✅ Loaded OpenCV Haar Cascade face detector (fallback)")
except Exception as e:
    print(f"⚠️ Could not load face detector: {e}")
    face_detector = None

print(f"✅ Loaded face recognition model: {MODEL_PATH}")
print(f"📋 Model trained with classes: {model.names}")


def detect_and_crop_face(image_bgr):
    """
    Use OpenCV to detect face and return cropped face region.
    Returns cropped face image or None if no face detected.
    """
    h, w, _ = image_bgr.shape
    
    if face_detector is None:
        return None
    
    # Try DNN detector first
    if isinstance(face_detector, cv2.dnn_Net):
        # DNN detector expects blob input
        blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300), [104, 117, 123])
        face_detector.setInput(blob)
        detections = face_detector.forward()
        
        # Find the best detection (highest confidence)
        best_conf = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_conf:  # min confidence 0.5
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                best_box = (x1, y1, x2, y2)
                best_conf = confidence
        
        if best_box:
            x1, y1, x2, y2 = best_box
    else:
        # Use Haar Cascade
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, width, height = largest_face
            x1, y1, x2, y2 = x, y, x + width, y + height
        else:
            return None
    
    # Add padding (20% on each side)
    width = x2 - x1
    height = y2 - y1
    padding_x = int(width * 0.2)
    padding_y = int(height * 0.2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)
    
    # Crop the face region
    cropped_face = image_bgr[y1:y2, x1:x2]
    
    print(f"     🎯 Face detected: bbox=({x1},{y1}) to ({x2},{y2}), size={cropped_face.shape}")
    
    return cropped_face


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

        # Try multiple orientations (phone cameras sometimes send rotated images)
        # Orientation 1: Original
        # Orientation 2: Rotated 90° clockwise (portrait mode)
        # Orientation 3: Rotated 90° counter-clockwise
        # Orientation 4: Rotated 180° (upside down)
        images_to_try = [
            img_bgr,
            cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(img_bgr, cv2.ROTATE_180),
        ]

        best_results = []
        found_match = False

        print(f"\n🔍 Verifying NFC ID: '{nfc_id}'")
        print(f"🔄 Will try 4 orientations (phone cameras sometimes send rotated images)")

        for i, current_img in enumerate(images_to_try):
            orientation_names = ["Original", "Rotated 90° CW", "Rotated 90° CCW", "Rotated 180°"]
            print(f"  👉 Trying orientation {i + 1}/4: {orientation_names[i]}...")
            
            # Step 1: Use OpenCV to detect and crop face
            cropped_face = detect_and_crop_face(current_img)
            
            # Step 2: If face detected, use cropped face; otherwise use full image
            if cropped_face is not None and cropped_face.size > 0:
                print(f"     ✅ Using cropped face for YOLO inference (original size: {cropped_face.shape})")
                face_img = cropped_face
            else:
                print(f"     ⚠️ No face detected by OpenCV, using full image")
                face_img = current_img
            
            # Step 3: Resize image to 640x640 (YOLO's optimal input size for best accuracy)
            # This is crucial - YOLO models are trained on 640x640 images
            original_size = face_img.shape[:2]
            face_img_resized = cv2.resize(face_img, (640, 640), interpolation=cv2.INTER_LINEAR)
            print(f"     🔄 Resized image from {original_size[1]}x{original_size[0]} to 640x640 (YOLO optimal size)")
            
            # Step 4: Run YOLO on the resized image
            # Using confidence threshold of 0.2 (balance between accuracy and detection)
            # Lower values (0.1) catch more but may have false positives
            # Higher values (0.25-0.3) are more strict but may miss some detections
            results = model(face_img_resized, verbose=False, conf=0.2)
            
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
                        # Prioritize this result (highest confidence match first)
                        best_results = [detection] + [r for r in best_results if r != detection]
                        print(f"     ✅ MATCH FOUND! '{class_name}' == '{nfc_id}' (confidence: {confidence:.3f})")
                        # Don't break - continue to find all matches, but we'll prioritize this one

            if found_match:
                print(f"  ✅ Match found in orientation {i + 1} ({orientation_names[i]})!")
                # Continue to check other orientations for higher confidence matches
                # but we already have a match, so we can break if confidence is high enough
                if any(r["confidence"] > 0.5 and r["class"].lower() == nfc_id.lower() for r in best_results):
                    print(f"  🎯 High confidence match found, stopping orientation search")
                    break

        # Remove duplicates and sort by confidence (highest first)
        # Keep the highest confidence detection for each class
        class_dict = {}
        for res in best_results:
            class_name = res["class"]
            if class_name not in class_dict or res["confidence"] > class_dict[class_name]["confidence"]:
                class_dict[class_name] = res
        
        # Sort by confidence (highest first), but prioritize matches with NFC ID
        unique_results = list(class_dict.values())
        unique_results.sort(key=lambda x: (
            x["class"].lower() != nfc_id.lower(),  # Matches first
            -x["confidence"]  # Then by confidence (descending)
        ))

        print(f"📊 Final Results: {unique_results}")
        
        # Check if we have a match
        has_match = any(r["class"].lower() == nfc_id.lower() for r in unique_results)
        if has_match:
            match_result = next(r for r in unique_results if r["class"].lower() == nfc_id.lower())
            print(f"✅ FINAL MATCH: '{match_result['class']}' with confidence {match_result['confidence']:.3f}")
        else:
            print(f"❌ NO MATCH: NFC ID '{nfc_id}' not found in detections")

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


