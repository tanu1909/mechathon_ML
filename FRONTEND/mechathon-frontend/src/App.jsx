import React, { useState } from "react";
import "./App.css";
import axios from 'axios'

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null); 
  const [result, setResult] = useState("");  //predicted value

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch("http://127.0.0.1:5002/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      console.log(err);
      alert("Error predicting. Check backend!");
    }
  };

  return (
    <div className="container">
      <h2>Image Upload & Prediction</h2>

      <input type="file" onChange={handleImageUpload} />

      {preview && (
        <img src={preview} alt="preview" className="preview" />
      )}

      <button onClick={handlePredict}>Predict</button>

      {result && <h3>Prediction: {result}</h3>}
    </div>
  );
}

export default App;
