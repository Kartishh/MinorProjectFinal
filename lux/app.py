from flask import Flask, render_template, request, url_for
import os
from PIL import Image
import numpy as np
from skimage.measure import shannon_entropy

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to calculate entropy and recommend model
def recommend_model_by_complexity(image_path, entropy_threshold=6.5):
    img = Image.open(image_path).convert("L")
    entropy_value = float(shannon_entropy(np.array(img)))
    model = "Teacher Model" if entropy_value > entropy_threshold else "Student Model"
    return entropy_value, model

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files.get("image")
        if image:
            # Save uploaded image
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            # Calculate entropy and model
            entropy, model = recommend_model_by_complexity(image_path)

            # URL for template
            image_url = url_for('static', filename=f"uploads/{image.filename}")

            return render_template(
                "index.html",
                result=True,
                entropy=entropy,
                model=model,
                image_url=image_url
            )

    return render_template("index.html", result=False)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6543, debug=True)
