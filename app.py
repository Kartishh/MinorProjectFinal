from flask import Flask, render_template, request, url_for, abort
import os
import time
from uuid import uuid4
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import numpy as np
from skimage.filters import sobel
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

# Optional: Gemini explainability
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

app = Flask(__name__)

# Config
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load .env and configure Gemini
if _HAS_GEMINI:
    try:
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        else:
            _HAS_GEMINI = False
    except Exception:
        _HAS_GEMINI = False

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_features(image_path: str) -> dict:
    """Compute multiple technical features for image complexity. Make it as much detailed as possible. All the details below should be explained in depth.

    Returns a dict with:
      - edge_density: proportion of strong gradients via Sobel
      - glcm_contrast: GLCM contrast (texture complexity)
      - glcm_entropy: entropy over normalized GLCM
      - hf_energy_ratio: high-frequency energy ratio in FFT domain
      - color_variance: variance across RGB channels (0 if grayscale)
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
        gray = rgb2gray(rgb)

    # Edge density via Sobel magnitude
    sob = np.abs(sobel(gray))
    threshold = np.percentile(sob, 75)
    edge_density = float((sob > threshold).mean())

    # Texture via GLCM (coarse quantization to 8-bit)
    gray_ubyte = img_as_ubyte(gray)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_ubyte, distances=distances, angles=angles, symmetric=True, normed=True)
    glcm_contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    # GLCM entropy
    p = glcm.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log2(p, where=(p > 0))
        glcm_entropy = float(-np.sum(p * logp))

    # Frequency domain: high-frequency energy ratio
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = 0.25 * max(h, w)
    hf_mask = radius >= cutoff
    hf_energy = float((magnitude[hf_mask] ** 2).sum())
    total_energy = float((magnitude ** 2).sum()) + 1e-9
    hf_energy_ratio = hf_energy / total_energy

    # Color variance (if color image)
    color_variance = float(np.var(rgb.astype(np.float32) / 255.0))

    return {
        'edge_density': edge_density,
        'glcm_contrast': glcm_contrast,
        'glcm_entropy': glcm_entropy,
        'hf_energy_ratio': hf_energy_ratio,
        'color_variance': color_variance,
    }


def compute_complexity_score(features: dict) -> float:
    """Combine normalized features into an overall complexity score [0, 1+]."""
    # Rough normalizations to bring features to comparable scales
    nd = lambda x, s: min(1.0, x / s)
    edge = nd(features['edge_density'], 0.25)           # typical 0-0.25
    contrast = nd(features['glcm_contrast'], 200.0)     # depends on image; cap at 200
    tentropy = nd(features['glcm_entropy'], 20.0)       # entropy of GLCM; cap
    hf = nd(features['hf_energy_ratio'], 0.4)           # typical 0-0.4
    colvar = nd(features['color_variance'], 0.05)       # small range for normalized images

    # Weighted sum (tunable)
    score = (
        0.25 * edge +
        0.25 * hf +
        0.20 * contrast +
        0.20 * tentropy +
        0.10 * colvar
    )
    return float(score)


def recommend_model_by_features(image_path: str, threshold: float = 0.6):
    feats = compute_features(image_path)
    score = compute_complexity_score(feats)
    model = "Teacher Model" if score >= threshold else "Student Model"
    return feats, score, model, threshold


def generate_explanation(features: dict, score: float, model: str, threshold: float) -> str:
    base_reason = (
        f"Composite score={score:.3f} (threshold {threshold}). Recommended: {model}. "
        f"Key signals — edges: {features['edge_density']:.3f}, texture contrast: {features['glcm_contrast']:.1f}, "
        f"GLCM entropy: {features['glcm_entropy']:.1f}, high‑freq ratio: {features['hf_energy_ratio']:.3f}, "
        f"color variance: {features['color_variance']:.4f}."
    )

    if not _HAS_GEMINI:
        if model == "Teacher Model":
            return (
                base_reason + " High edge density, rich textures, and stronger high‑frequency content point to complex details, so the Teacher model is preferred for fidelity."
            )
        return (
            base_reason + " Lower structural/textural complexity suggests the Student model will be efficient without noticeable quality loss."
        )

    try:
        prompt = (
            "You are an expert ML assistant explaining a classification that chooses between two models: "
            "Teacher (quality-first) vs Student (efficiency-first). Use the following image features to justify the choice. "
            "Explain in 3-5 concise sentences, non-jargon and user-friendly, but technically grounded. "
            "Focus on how edges, texture (GLCM), high-frequency energy, and color variance relate to visual complexity. "
            "Avoid speculation beyond the features.\n\n"
            f"Features:\n"
            f"- Edge density: {features['edge_density']:.3f}\n"
            f"- GLCM contrast: {features['glcm_contrast']:.1f}\n"
            f"- GLCM entropy: {features['glcm_entropy']:.1f}\n"
            f"- High-frequency energy ratio: {features['hf_energy_ratio']:.3f}\n"
            f"- Color variance: {features['color_variance']:.4f}\n"
            f"Composite score: {score:.3f}\nThreshold: {threshold}\nSelected model: {model}"
        )
        model_client = genai.GenerativeModel("gemini-1.5-flash")
        response = model_client.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass
    return base_reason

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    # Threshold for composite score in [0, ~1]; tune as needed
    threshold = 0.6
    if request.method == "POST":
        image = request.files.get("image")

        if not image or image.filename == "":
            abort(400, description="No image provided.")
        if not allowed_file(image.filename):
            abort(400, description="Unsupported file type.")

        # Save uploaded image with safe, unique name
        try:
            original = secure_filename(image.filename)
            name, ext = os.path.splitext(original)
            unique = f"{int(time.time())}-{uuid4().hex[:8]}{ext.lower()}"
            save_name = unique
            image_path = os.path.join(UPLOAD_FOLDER, save_name)
            image.save(image_path)

            # Validate image actually opens
            with Image.open(image_path) as _:
                pass
        except UnidentifiedImageError:
            abort(400, description="Uploaded file is not a valid image.")
        except Exception:
            abort(500, description="Failed to process the image.")

        # Compute features and recommendation
        features, score, model, threshold = recommend_model_by_features(image_path, threshold=threshold)
        explanation = generate_explanation(features, score, model, threshold)

        image_url = url_for('static', filename=f"uploads/{save_name}")
        # Map score [0,1] to 0-100 bar
        score_bar = max(0, min(100, int(score * 100)))

        return render_template(
            "index.html",
            result=True,
            entropy=score,  # kept key for template compatibility; now represents composite score
            model=model,
            image_url=image_url,
            entropy_bar=score_bar,
            threshold=threshold,
            explanation=explanation,
        )

    return render_template("index.html", result=False, threshold=threshold)

# Run app
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=6543, debug=debug)

