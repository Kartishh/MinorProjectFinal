# Image Complexity Model Recommender

A simple Flask web application that allows users to upload images, calculates the image complexity using **Shannon entropy**, and recommends a model based on the complexity.

---

## Features

- Upload images via a web interface
- Calculate **entropy** of the image
- Recommend a model:
  - **Teacher Model** for high-complexity images
  - **Student Model** for low-complexity images
- Preview uploaded images in the browser

---



Install dependencies using:

```bash
pip install numpy scikit-image pillow
````

`requirements.txt` example:

```
Flask
Pillow
numpy
scikit-image
```

---

## Project Structure

```
difussion/
├── app.py
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

6. Upload an image and see the recommended model based on its complexity.

---

## How It Works

* The app converts the uploaded image to grayscale
* Computes **Shannon entropy** to measure complexity
* Uses a threshold (`6.5` by default) to recommend:

  * **Teacher Model** → High complexity
  * **Student Model** → Low complexity

---



## Author

[FardeenSK004](https://github.com/FardeenSK004)


