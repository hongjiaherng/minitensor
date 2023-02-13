import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { computeStrides } from "../../utils";
import { asStrided, asStrided_ } from "../creation/as_strided";

export function reshape<D extends DType>(
  x: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // create new tensor with same storage, modify shape, strides, and offset (no change)
  shape = inferUnknownDimension(x.size, shape);
  return asStrided(x, shape, computeStrides(shape));
}

export function reshape_<D extends DType>(
  x: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // modify shape, strides, and offset (no change)
  shape = inferUnknownDimension(x.size, shape);
  return asStrided_(x, shape, computeStrides(shape));
}

export function inferUnknownDimension(size: number, shape: number[]): number[] {
  // infer unknown dimension from size
  // if no unknown dimension, return shape
  // if more than one unknown dimension, throw error

  // count number of unknown dimensions
  const unknownDim = shape.reduce((total, dim) => {
    if (dim === 0)
      throw new Error(`Shape [${shape}] is invalid for input of size ${size}`);
    if (total > 1)
      throw new Error(
        "Cannot infer shape with more than one unknown dimension"
      );
    return total + (dim === -1 ? 1 : 0);
  }, 0);
  if (unknownDim === 0) return shape;

  // calculate size of known dimensions
  const knownSize = shape.reduce((prev, dim) => {
    return prev * (dim === -1 ? 1 : dim);
  }, 1);

  // calculate size of unknown dimension
  if (size % knownSize !== 0)
    throw new Error(`Shape [${shape}] is invalid for input of size ${size}`);
  const unknownSize = size / knownSize;

  return shape.map((dim) => {
    return dim === -1 ? unknownSize : dim;
  });
}
