# Image Complexity Model Recommender

A polished Flask application to recommend Teacher vs Student models from a single image input. It computes a multi-feature **complexity score** (edges, texture via GLCM, frequency content, color variance) and provides a short **explainable AI** rationale using Gemini if configured.

---

## Features

- Upload images via a web interface
- Calculate a composite complexity score and visualize it with a bar
- Recommend a model and show a compact explanation
- Gemini explainability (optional, via `.env`)

---



### Setup

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) (Optional) Configure Gemini for explanations

Create a `.env` file next to `app.py` with your API key:

```
GEMINI_API_KEY=your_api_key_here
```

Without this key, the app still runs and returns a deterministic explanation.

---

## Project Structure

```
lux/
├── app.py
├── requirements.txt
├── static/
│   ├── styles.css
│   └── uploads/  # Uploaded images
├── templates/
│   └── index.html
└── README.md
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/FardeenSK004/lux.git
cd difussion
```

2. (Optional) Create a virtual environment:

```bash
python -m venv stude
source stude/bin/activate  # On Linux/macOS
stude\Scripts\activate     # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open your browser and go to:

```
http://localhost:6543
```

6. Upload an image. The app auto-selects the model based on the computed complexity score.

---

## How It Works

* Converts the uploaded image to RGB + grayscale
* Computes technical features:
  * Edge density (Sobel)
  * Texture metrics (GLCM contrast, GLCM entropy)
  * High-frequency energy ratio (FFT)
  * Color variance (RGB)
* Normalizes and combines features into a composite complexity score (0–1)
* Uses a default threshold (`0.6`) to recommend Teacher (complex) vs Student (simple)

### Explainability
If `GEMINI_API_KEY` is set, the app asks Gemini for a short, technical-yet-accessible explanation grounded in the computed features. Otherwise it returns a concise local rationale.

### Integration Hooks (for your main project)
This app outputs the recommended model name and the composite complexity score. To integrate with your existing Teacher/Student models, wire the selected model into your inference code where indicated in `app.py` (search for `recommend_model_by_features`). You can:

- Read `model` and `entropy` (now the composite score) from the POST result and dispatch to your inference function
- Or embed this Flask app as a microservice that returns the decision and metadata

---



## Author

[FardeenSK004](https://github.com/FardeenSK004)


