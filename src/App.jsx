import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

import "./App.css";

function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser
      .fromPixels(img)
      .resizeNearestNeighbor([256, 256], true)
      .div(255)
      .reshape([1, 256, 256, 3]);
    return tensor;
  });
}

export default function App() {
  const [model, setModel] = useState(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      const modelPath = "/localization/tfjs_quant_uint8/model.json";
      const loadedModel = await tf.loadGraphModel(modelPath);
      setModel(loadedModel);

      console.log("Model Loaded successfully.");
    };

    loadModel();
  }, []);

  const drawBoundingBox = () => {
    if (!model) return;
    const inputTensor = preprocessImage(imgRef.current);
    const predictions = model.predict(inputTensor);
    predictions.data().then((box) => {
      const canvas = canvasRef.current;
      const imgWidth = imgRef.current.width;
      const imgHeight = imgRef.current.height;
      canvas.width = imgWidth;
      canvas.height = imgHeight;

      const ctx = canvas.getContext("2d");

      const startX = box[0] * imgWidth;
      const startY = box[1] * imgHeight;

      const width = (box[2] - box[0]) * imgWidth;
      const height = (box[3] - box[1]) * imgHeight;

      ctx.strokeStyle = "#0F0";
      ctx.lineWidth = 4;

      ctx.strokeRect(startX, startY, width, height);
    });
  };

  return (
    <div className='container'>
      <h1>InceptionV3 Classification</h1>
      <img
        ref={imgRef}
        src='/images/cat.jpg'
        alt='Test'
        crossOrigin='anonymous'
        style={{ width: 299, height: 299 }}
      />
      <button onClick={drawBoundingBox}>Bounding Box</button>

      <canvas className='space' ref={canvasRef}></canvas>
    </div>
  );
}
