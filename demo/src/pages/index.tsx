import type { InferGetStaticPropsType } from "next";
import { useState } from "react";
import * as minitensor from "minitensor";

export const getStaticProps = async () => {
  const typedArray = Float32Array.from([1, 2, 3, 4]);
  console.log("typedArray", typedArray);
  const data = {
    typedArray: Array.from(typedArray),
    minitensor: minitensor.float32
  };
  console.log("data", data);
  return {
    props: {
      data
    }
  };
};

const Home = ({ data }: InferGetStaticPropsType<typeof getStaticProps>) => {
  const [typedArray] = useState(Float32Array.from([1, 2, 3, 4]));
  const [tensor] = useState(minitensor.tensor([1, 2, 3, 4]));
  return (
    <>
      <div>{JSON.stringify(data)}</div>
      <div>{JSON.stringify(typedArray)}</div>
      <div>{minitensor.float32}</div>
      <div>{JSON.stringify(tensor)}</div>
      <div>{JSON.stringify(minitensor.computeStrides([3, 3]))}</div>
      <div>{JSON.stringify(tensor.expand([4, 4]))}</div>
      {/* <div>{JSON.stringify(minitensor.datasets.makeBlobs(10, 2, 3))}</div> */}
      {/* <div>{minitensor.div}</div> */}
    </>
  );
};

export default Home;
