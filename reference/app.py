from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort
import io
import os
import base64

app = FastAPI(title="ArcFace Backend",
              description="MediaPipe face detection + ArcFace embedding + matching API",
              version="0.1")

# Allow CORS from frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for test UI
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount('/static', StaticFiles(directory=static_dir), name='static')

@app.get('/')
async def root():
    return RedirectResponse(url='/static/test.html')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# Note: do not create a global FaceMesh instance (not thread-safe).
# We'll create a per-request FaceMesh inside the handler using static_image_mode=True.

# Load ArcFace ONNX model (support env var or common repo locations)
model_env = os.getenv('ARC_FACE_ONNX_PATH')
# Common fallback locations to search for the ONNX file
default_paths = [
    os.path.join(os.path.dirname(__file__), 'models', 'arcface.onnx'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'arcface.onnx'),
    os.path.join(os.getcwd(), 'backend', 'models', 'arcface.onnx'),
    os.path.join(os.getcwd(), 'models', 'arcface.onnx'),
    os.path.join(os.getcwd(), 'arcface.onnx'),
]

onnx_path = None

if model_env:
    if os.path.exists(model_env) and model_env.lower().endswith('.onnx'):
        onnx_path = model_env
        print(f"🔧 Using ONNX model from ARC_FACE_ONNX_PATH: {onnx_path}")
    else:
        print(f"⚠️ ARC_FACE_ONNX_PATH is set but file not found or not .onnx: {model_env}")

if not onnx_path:
    for p in default_paths:
        if os.path.exists(p) and p.lower().endswith('.onnx'):
            onnx_path = p
            break

ort_session = None
input_name = None

if onnx_path:
    try:
        print(f"🔍 Attempting to load ONNX model from: {onnx_path}")
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
        print(f"✅ Loaded ONNX model from: {onnx_path}")
    except Exception as e:
        print("❌ Failed to load ONNX model:", e)
        ort_session = None
else:
    print("⚠️ No ONNX model found in default locations. Embedding endpoint will fail until an ONNX model is available.")


class MatchRequest(BaseModel):
    embedding1: list[float]
    embedding2: list[float]


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """Accepts an image file, runs MediaPipe face mesh to detect face, crops & preprocesses and runs ArcFace ONNX model to return a 512-d embedding."""
    if ort_session is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded on server")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face mesh using a per-request FaceMesh (static_image_mode=True)
    # This avoids cross-request state and produces consistent detections for single images.
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return {"success": False, "faceDetected": False, "error": "No faces detected"}

    # Use first face's landmarks to compute bounding box
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    xs = [min(max(int(lm.x * w), 0), w - 1) for lm in landmarks]
    ys = [min(max(int(lm.y * h), 0), h - 1) for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Expand box slightly
    pad_x = int((x_max - x_min) * 0.2)
    pad_y = int((y_max - y_min) * 0.3)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # Prepare debug bbox and landmarks (pixel coordinates)
    debug = {
        'bbox': [int(x1), int(y1), int(x2), int(y2)],
        'landmarks': [{'x': int(lm.x * w), 'y': int(lm.y * h)} for lm in landmarks]
    }

    face_crop = img_rgb[y1:y2, x1:x2]

    if face_crop.size == 0:
        return {"success": False, "faceDetected": False, "error": "Face crop failed"}

    # Preprocess to 112x112, normalize as (img - 127.5)/128.0
    face_resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
    face_float = face_resized.astype(np.float32)
    preprocessed = (face_float - 127.5) / 128.0

    # Determine model expected input layout (NHWC vs NCHW) and arrange accordingly
    # debug dict will be appended further with model input details
    try:
        input_meta = ort_session.get_inputs()[0]
        input_shape = input_meta.shape
        debug['modelInputShape'] = input_shape

        # Replace None with 1 for simple checking
        resolved_shape = [s if isinstance(s, int) else 1 for s in input_shape]

        arranged = 'unknown'
        if len(resolved_shape) >= 4 and resolved_shape[1] == 3:
            # Model expects NCHW (1,3,112,112)
            input_tensor = np.transpose(preprocessed, (2, 0, 1))[None, ...].astype(np.float32)
            arranged = 'NCHW'
        elif len(resolved_shape) >= 4 and resolved_shape[-1] == 3:
            # Model expects NHWC (1,112,112,3)
            input_tensor = preprocessed[None, ...].astype(np.float32)
            arranged = 'NHWC'
        else:
            # Fallback to NHWC
            input_tensor = preprocessed[None, ...].astype(np.float32)
            arranged = 'assumed_NHWC'

        debug['arrangedInputLayout'] = arranged
        # Provide a small fingerprint of input tensor for debugging
        input_flat = input_tensor.flatten()
        debug['inputSample'] = input_flat[:10].astype(float).tolist()
        debug['inputShapeUsed'] = list(input_tensor.shape)

        # ALSO include original (pre-normalization) pixel values for the same sample positions
        try:
            # Convert back: original = value*128 + 127.5
            orig_samples = ((input_flat[:10] * 128.0) + 127.5).astype(int).tolist()
            debug['inputOriginalSample'] = orig_samples
        except Exception:
            debug['inputOriginalSample'] = None

        # Add a small base64 preview of the face crop (112x112) so UI can show what was fed to model
        try:
            # face_resized is RGB uint8 array; convert to BGR for imencode
            preview_bgr = cv2.cvtColor(face_resized.astype(np.uint8), cv2.COLOR_RGB2BGR)
            success, img_buf = cv2.imencode('.png', preview_bgr)
            if success:
                b64 = base64.b64encode(img_buf.tobytes()).decode('ascii')
                debug['faceCropPreview'] = f"data:image/png;base64,{b64}"
            else:
                debug['faceCropPreview'] = None
        except Exception as e:
            debug['faceCropPreviewError'] = str(e)

    except Exception as e:
        debug['inputDetectionError'] = str(e)
        # Fallback to NHWC
        input_tensor = preprocessed[None, ...].astype(np.float32)
        debug['arrangedInputLayout'] = 'fallback_NHWC'
        debug['inputShapeUsed'] = list(input_tensor.shape)

    try:
        outputs = ort_session.run(None, {input_name: input_tensor})
        emb = outputs[0].reshape(-1)
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        emb_list = emb.astype(float).tolist()

        return {
            "success": True,
            "embedding": emb_list,
            "faceDetected": True,
            "confidence": 1.0,
            "processingTime": 0,
            "debug": debug
        }
    except Exception as e:
        # Include debug info in error for UI visibility
        raise HTTPException(status_code=500, detail={"error": f"Model inference failed: {e}", "debug": debug})


@app.post("/match")
async def match_embeddings(req: MatchRequest):
    a = np.array(req.embedding1, dtype=np.float32)
    b = np.array(req.embedding2, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        raise HTTPException(status_code=400, detail="Embeddings must be non-empty and of same shape")

    dot = float(np.dot(a, b))
    n1 = float(np.linalg.norm(a))
    n2 = float(np.linalg.norm(b))
    if n1 == 0 or n2 == 0:
        similarity = 0.0
    else:
        similarity = dot / (n1 * n2)

    return {"similarity": float(similarity), "dot": dot, "norm1": n1, "norm2": n2}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
