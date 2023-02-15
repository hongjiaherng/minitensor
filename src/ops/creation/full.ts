import { Tensor } from "../../tensor";
import { DType, PrimTypeMap } from "../../types";
import { tensor } from "./tensor";

export function full<D extends DType>(
  shape: number[],
  value: PrimTypeMap[D],
  dtype: D = DType.float32 as D
): Tensor<D> {
  const data = Array(shape.reduce((a, b) => a * b, 1)).fill(value);
  dtype = typeof value === "boolean" ? (DType.bool as D) : dtype;
  return tensor(data, shape, dtype);
}
