import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function App() {
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await tf.loadLayersModel("/ttt-model/ttt_model.json");
      setModel(loadedModel);

      console.log("Tic-Tac-Toe model loaded!");
    };

    loadModel();
  }, []);

  const predictWinner = () => {
    if (!model) return "Model not loaded yet";
    const emptyBoard = tf.zeros([9]);

    const betterBlockMe = tf.tensor([-1, 0, 0, 1, 1, -1, 0, 0, -1]);
    const goForTheKill = tf.tensor([1, 0, 1, 0, -1, -1, -1, 0, 1]);

    const matches = tf.stack([emptyBoard, betterBlockMe, goForTheKill]);

    const result = model.predict(matches);

    result.reshape([3, 3, 3]).print();
  };

  return (
    <div>
      <h1>Tic-Tac-Toe Predictor</h1>
      <button
        onClick={() => {
          predictWinner();
        }}
      >
        Predict Winner
      </button>
    </div>
  );
}
