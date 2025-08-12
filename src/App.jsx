import { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { INCEPTION_CLASSES } from "./inception_classes";

function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser
      .fromPixels(img)
      .resizeBilinear([299, 299], true)
      .div(255)
      .reshape([1, 299, 299, 3]);
    return tensor;
  });
}

export default function App() {
  const [model, setModel] = useState(null);
  const imgRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      const modelPath = "/inceptionV3/model.json";
      const loadedModel = await tf.loadGraphModel(modelPath);
      setModel(loadedModel);

      console.log("Model Loaded successfully.");
    };

    loadModel();
  }, []);

  const handlePredict = () => {
    if (!model) return;
    const inputTensor = preprocessImage(imgRef.current);
    const predictions = model.predict(inputTensor);
    predictions.data().then((arr) => {
      const { indices } = tf.topk(arr, 3);

      const prediction_indices = indices.dataSync();

      console.log(`
        First Prediction ${INCEPTION_CLASSES[prediction_indices[0]]},
        Second Prediction ${INCEPTION_CLASSES[prediction_indices[1]]},
        Third Third Prediction ${INCEPTION_CLASSES[prediction_indices[2]]}
        `);

      inputTensor.dispose();
      model.dispose();
    });
  };

  return (
    <div>
      <h1>InceptionV3 Classification</h1>
      <img
        ref={imgRef}
        src='/images/cat.jpg'
        alt='Test'
        crossOrigin='anonymous'
        style={{ width: 299, height: 299 }}
      />
      <button onClick={handlePredict}>Classify</button>
    </div>
  );
}
