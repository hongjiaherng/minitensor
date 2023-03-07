import {
  computeStrides, inferUnknownDimension
} from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { isContiguous } from "../../tensor_util";
import { DType } from "../../types";
import { asStrided_, tensor } from "../creation";
import { view } from "./view";

export function reshape<D extends DType>(
  x: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // create new tensor with same storage, modify shape, strides, and offset (no change)

  // check if x is contiguous, proceed shape creation
  // otherwise, create a new tensor with same data, but new shape, strides, and offset (make it contiguous)

  try {
    return view(x, shape);
  } catch (error) {
    shape = inferUnknownDimension(x.size, shape);
    return tensor(x.array(), shape, x.dtype) as Tensor<D>;
  }
}

export function reshape_<D extends DType>(
  x: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // modify shape, strides, and offset (no change)

  // check if x is contiguous, proceed shape creation
  // otherwise, raise error, as it is not possible to modify shape and strides of a non-contiguous tensor in-place
  // To reshape a non-contiguous tensor, a new tensor must be created with new storage
  shape = inferUnknownDimension(x.size, shape);
  if (isContiguous(x)) return asStrided_(x, shape, computeStrides(shape));
  throw new Error(
    "Cannot reshape a non-contiguous tensor in-place. Use reshape() instead."
  );
}
