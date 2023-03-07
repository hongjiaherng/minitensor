import type { GetStaticProps, InferGetStaticPropsType, NextPage } from "next";
import ScatterPlot from "../components/ScatterPlot";
import * as mt from "minitensor";
import { useEffect, useRef, useState } from "react";

export type Dataset = {
  X: mt.Tensor<mt.DType.float32 | mt.DType.float64>;
  y: mt.Tensor<mt.DType.float32 | mt.DType.float64>;
};

const generateDataset = (): Dataset => {
  return mt.datasets.makeBlobs(1000, 2, 3, 1.0, [-10, 10]);
};

const Home: NextPage = () => {
  const [dataset, setDataset] = useState<Dataset>();
  const [isPaused, setIsPaused] = useState<boolean>(true);
  const interval = useRef<NodeJS.Timeout | null>();

  useEffect(() => {
    if (!isPaused) {
      interval.current = setInterval(() => setDataset(generateDataset()), 1000);
    } else {
      clearInterval(interval.current!);
      interval.current = null;
    }

    return () => clearInterval(interval.current!);
  }, [isPaused]);

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="flex-col">
        <div className="text-center text-4xl font-bold p-2">minitensor</div>
        {dataset && <ScatterPlot data={dataset} width={500} height={500} />}
        <div className="flex justify-center p-2 space-x-2">
          <button
            className="w-40 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            onClick={() => setIsPaused(false)}
          >
            Generate
          </button>
          <button
            className="w-40 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            onClick={() => setIsPaused(true)}
          >
            Stop
          </button>
          <button
            className="w-40 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            onClick={() => setDataset(generateDataset())}
          >
            Generate Once
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
