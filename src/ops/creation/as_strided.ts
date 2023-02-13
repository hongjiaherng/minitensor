import { Tensor } from "../../tensor";
import { DType } from "../../types";

export function asStrided<D extends DType>(
  x: Tensor<D>,
  shape: number[],
  strides: number[],
  offset: number = x.offset
): Tensor<D> {
  // create new tensor with same storage, modify shape, strides, and offset (no change)
  return new Tensor(x.data, shape, strides, offset);
}

export function asStrided_<D extends DType>(
  x: Tensor<D>,
  shape: number[],
  strides: number[],
  offset: number = x.offset
): Tensor<D> {
  // In-place version of asStrided
  x.shape = shape;
  x.strides = strides;
  x.offset = offset;
  x.size = shape.reduce((a, b) => a * b, 1);
  return x;
}
