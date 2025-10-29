# AI Model Recommender - Image Complexity Classification

A polished Flask application to recommend Teacher vs Student models from a single image input. It computes a multi-feature **complexity score** using advanced computer vision techniques (edges via Sobel filters, texture via GLCM, frequency content via FFT, color variance) and provides intelligent **explainable AI** rationale using Gemini if configured.

---

## Features

- Upload images via a modern web interface
- Multi-feature complexity analysis (edge density, texture metrics, frequency analysis, color variance)
- Calculate a composite complexity score and visualize it with a progress bar
- Intelligent model recommendation (Teacher for complex images, Student for simpler ones)
- Explainable AI explanations (Gemini-powered or deterministic fallback)

---

## Setup

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini for enhanced explanations

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)

Without this key, the app still runs and returns a deterministic explanation based on computed features.

---

## Project Structure

```
MinorProjectFinal/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── static/
│   ├── styles.css        # Modern dark theme styles
│   └── uploads/          # Uploaded images directory
└── templates/
    └── index.html        # Main web interface
```

---

## Usage

1. Run the app:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:6543
```

3. Upload an image (PNG, JPG, JPEG, max 10 MB). The app automatically selects the appropriate model based on the computed complexity score.

---

## How It Works

The application uses a sophisticated multi-feature approach to classify image complexity:

### Feature Extraction
1. **Edge Density** (Sobel Operator): Detects strong gradients and edge structures
   - Uses the Sobel filter to compute gradient magnitude
   - Measures the proportion of pixels with gradients above the 75th percentile
   - Higher values indicate more structural detail and edges

2. **Texture Metrics (GLCM)**:
   - **GLCM Contrast**: Measures local intensity variations (texture roughness)
   - **GLCM Entropy**: Quantifies texture randomness and information content
   - Computed across multiple distances [1, 2, 4] and angles [0°, 45°, 90°, 135°]

3. **High-Frequency Energy Ratio (FFT)**:
   - Performs 2D Fast Fourier Transform on the grayscale image
   - Identifies high-frequency components (fine details, textures, noise)
   - Calculates the ratio of high-frequency energy to total energy
   - Cutoff set at 25% of the maximum image dimension

4. **Color Variance**:
   - Measures variance across RGB channels when normalized
   - Indicates color diversity and complexity

### Complexity Score Calculation
- All features are normalized to comparable scales (0-1)
- Weighted combination:
  - Edge density: 25%
  - High-frequency energy: 25%
  - GLCM contrast: 20%
  - GLCM entropy: 20%
  - Color variance: 10%
- Default threshold: **0.6**
  - Score ≥ 0.6 → **Teacher Model** (quality-first)
  - Score < 0.6 → **Student Model** (efficiency-first)

### Explainability
If `GEMINI_API_KEY` is configured, the app uses Google's Gemini AI to generate a user-friendly, technically-grounded explanation of why a particular model was selected, referencing the specific feature values. Otherwise, it provides a concise deterministic explanation based on the computed metrics.

---

## Integration with Your Main Project

This application outputs the recommended model name and composite complexity score. To integrate with your existing Teacher/Student models:

1. The `recommend_model_by_features()` function in `app.py` returns the model recommendation
2. You can call this function directly from your inference pipeline
3. Or wrap this Flask app as a microservice that returns JSON with:
   - `model`: "Teacher Model" or "Student Model"
   - `complexity_score`: float (0-1)
   - `features`: dict with individual feature values
   - `explanation`: string

Example integration point:
```python
from app import recommend_model_by_features

features, score, model, threshold = recommend_model_by_features(image_path)
# Use `model` to route to your Teacher or Student inference function
```

---

## Technical Details

### Dependencies
- **Flask**: Web framework
- **Pillow**: Image processing
- **NumPy**: Numerical operations
- **scikit-image**: Advanced image analysis (Sobel, GLCM, FFT)
- **python-dotenv**: Environment variable management
- **google-generativeai**: Gemini API integration (optional)

### Supported Image Formats
- PNG
- JPEG/JPG
- Maximum file size: 10 MB

---

## Authors

- **Kartish** - [@Kartishh](https://github.com/Kartishh)
- **Lakshay Pahal** - [@LakshayPahal](https://github.com/LakshayPahal)

---

## License

This project is part of a minor project submission.

