import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { tensor } from "./tensor";

export function arange<D extends DType>(
  start: number,
  end: number,
  step: number = 1,
  dtype: D = DType.float32 as D
): Tensor<D> {
  const size = Math.floor((end - start) / step);
  const data = [];
  for (let i = 0; i < size; i++) {
    data.push(start + i * step);
  }
  return tensor(data, undefined, dtype);
}
