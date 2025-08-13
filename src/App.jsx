import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

import { CLASSES } from "./labels";

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
    console.log(result);

    const canvas = canvasRef.current;
    const imgWidth = imgRef.current.width;
    const imgHeight = imgRef.current.height;
    canvas.width = imgWidth;
    canvas.height = imgHeight;

    const detectionThreshold = 0.4;
    const iouThreshold = 0.5;
    const maxBoxes = 20;
    const prominentDetection = tf.topk(result[0]);
    const justBoxes = result[1].squeeze();
    const justValues = prominentDetection.values.squeeze();

    // Move results back to JavaScript in parallel
    const [maxIndices, scores, boxes] = await Promise.all([
      prominentDetection.indices.data(),
      justValues.array(),
      justBoxes.array(),
    ]);

    // https://arxiv.org/pdf/1704.04503.pdf, use Async to keep visuals
    const nmsDetections = await tf.image.nonMaxSuppressionWithScoreAsync(
      justBoxes, // Shape [numBoxes, 4]
      justValues, // Shape [numBoxes]
      maxBoxes, // Stop making boxes when this number is hit
      iouThreshold, // Allowed overlap value 0 to 1
      detectionThreshold, // Minimum detection score allowed
      1 // 0 is normal NMS, 1 is max Soft-NMS for overlapping support
    );
    const chosen = await nmsDetections.selectedIndices.data();

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, imgWidth, imgHeight);

    chosen.forEach((detection) => {
      ctx.strokeStyle = "#0F0";
      ctx.lineWidth = 4;
      ctx.fillStyle = "#0B0";
      ctx.font = "16px sans-serif";
      ctx.textBaseline = "top";
      const textHeight = 16;
      const textPad = 4;

      const detectedIndex = maxIndices[detection];
      const detectedClass = CLASSES[detectedIndex];
      const detectedScore = scores[detection];
      const dBox = boxes[detection];
      console.log(detectedClass, detectedScore);

      const label = `${detectedClass} ${Math.round(detectedScore * 100)}%`;
      const textWidth = ctx.measureText(label).width;

      // No negative values for start positions
      const startY = dBox[0] > 0 ? dBox[0] * imgHeight : 0;
      const startX = dBox[1] > 0 ? dBox[1] * imgWidth : 0;
      const height = (dBox[2] - dBox[0]) * imgHeight;
      const width = (dBox[3] - dBox[1]) * imgWidth;

      ctx.fillRect(startX, startY, textWidth + textPad, textHeight + textPad);

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(label, startX, startY);

      ctx.strokeRect(startX, startY, width, height);
    });

    tf.dispose([
      inputTensor,
      ...result,
      model,
      nmsDetections.selectedIndices,
      nmsDetections.selectedScores,
      prominentDetection.indices,
      prominentDetection.values,
      justBoxes,
      justValues,
    ]);

    console.log("Tensor Memory Status:", tf.memory().numTensors);
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
