# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allows React to communicate with Flask

# Load Model
model = tf.keras.models.load_model("mnist_cnn_advanced.keras")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if the file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']

    try:
        # 2. Open the image directly from bytes
        img = Image.open(file.stream)

        # 3. Preprocessing (Crucial Step!)
        # Convert to Grayscale ('L') and Resize to 28x28 to match MNIST
        img = img.convert('L').resize((28, 28))

        # Convert to numpy array
        arr = np.array(img)

        # 4. Invert Colors (Optional but recommended for MNIST)
        # MNIST is White Text on Black Background. 
        # If users upload Black Text on White Paper, we must invert.
        # This simple check inverts if the image is mostly bright.
        if np.mean(arr) > 127:
            arr = 255 - arr

        # 5. Reshape and Normalize
        arr = arr.reshape(1, 28, 28, 1)
        arr = arr.astype("float32") / 255.0

        # 6. Predict
        pred_probs = model.predict(arr)
        pred_label = np.argmax(pred_probs)
        confidence = float(np.max(pred_probs))

        return jsonify({
            "prediction": int(pred_label),
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002)