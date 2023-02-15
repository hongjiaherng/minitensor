import { Tensor } from "../../tensor";
import { DType, PrimTypeMap } from "../../types";
import { full } from "./full";

export function empty<D extends DType>(
  shape: number[],
  dtype: D = DType.float32 as D
): Tensor<D> {
  const value = dtype === DType.bool ? false : 0;
  return full(shape, value as PrimTypeMap[D], dtype);
}
