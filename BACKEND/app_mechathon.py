from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    arr = np.array([data["values"]])
    arr = arr.reshape(-1,28,28,1)
    pred = np.argmax(model.predict(arr))
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
