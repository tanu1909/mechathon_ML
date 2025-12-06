from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps 

app = Flask(__name__)
CORS(app)

# Make sure this filename matches your actual model file
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Check if the 'file' key is in the request (matches Frontend)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        #  Open the image and convert to Grayscale ('L')
        image = Image.open(file).convert('L')

        image = ImageOps.invert(image) 
        image = image.resize((28, 28))
        arr = np.array(image) / 255.0

        # Reshape to match model input (1 image, 28x28 pixels, 1 channel)
        arr = arr.reshape(1, 28, 28, 1)

 
        prediction = model.predict(arr)
        result = np.argmax(prediction)

        return jsonify({"prediction": int(result)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002)