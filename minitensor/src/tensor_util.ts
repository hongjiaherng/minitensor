import { Tensor } from "./tensor";
import { DType } from "./types";

export function sharesMemory<D extends DType>(
  x: Tensor<D>,
  y: Tensor<D>
): boolean {
  return x.data === y.data;
}

export function isContiguous<D extends DType>(x: Tensor<D>): boolean {
  /**
   *
   * A tensor is contiguous if:
   * - It occupies an unbroken block of memory, and
   * - no strides are 0
   *
   */

  // TODO: Implement isContiguous
  let stride = 1;

  for (let i = x.shape.length - 1; i >= 0; i--) {
    stride *= x.shape[i];
    if (x.strides[i - 1] !== undefined && x.strides[i - 1] !== stride)
      return false;
  }

  return true;
}
