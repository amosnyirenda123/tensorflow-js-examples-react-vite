import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

import "./App.css";

function preprocessImage(img) {
  return tf.tidy(() => {
    let singleBatch = tf.browser.fromPixels(img).expandDims(0);
    return singleBatch;
  });
}

export default function App() {
  const [model, setModel] = useState(null);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      const modelPath = "/ssd_mobilenet_v2/model.json";
      const loadedModel = await tf.loadGraphModel(modelPath);
      setModel(loadedModel);

      console.log("Model Loaded successfully.");
    };

    loadModel();
  }, []);

  const drawBoundingBoxes = async () => {
    if (!model) return;

    const inputTensor = preprocessImage(imgRef.current);
    const result = await model.executeAsync(inputTensor);

    const scores = await result[0].data();
    const boxes = await result[1].squeeze().array();

    const canvas = canvasRef.current;
    const imgWidth = imgRef.current.width;
    const imgHeight = imgRef.current.height;
    canvas.width = imgWidth;
    canvas.height = imgHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, imgWidth, imgHeight);

    boxes.forEach((box, i) => {
      if (scores[i] > 0.5) {
        ctx.strokeStyle = "#0F0";
        ctx.lineWidth = 2;
        const startY = box[0] * imgHeight;
        const startX = box[1] * imgWidth;
        const height = (box[2] - box[0]) * imgHeight;
        const width = (box[3] - box[1]) * imgWidth;
        ctx.strokeRect(startX, startY, width, height);
      }
    });

    tf.dispose([inputTensor, ...result]);
  };

  return (
    <div className='container'>
      <h1>SSD MOBILENET</h1>
      <img
        ref={imgRef}
        src='/images/cat.jpg'
        alt='Test'
        crossOrigin='anonymous'
        style={{ width: 299, height: 299 }}
      />
      <button onClick={() => drawBoundingBoxes()}>Draw Bounding</button>

      <canvas className='space' ref={canvasRef}></canvas>
    </div>
  );
}
