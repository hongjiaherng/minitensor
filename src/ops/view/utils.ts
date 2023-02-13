import { Tensor } from "../../tensor";
import { DType } from "../../types";

export function sharesMemory<D extends DType>(
  x: Tensor<D>,
  y: Tensor<D>
): boolean {
  return x.data === y.data;
}
