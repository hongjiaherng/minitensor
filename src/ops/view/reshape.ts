import {
  inferUnknownDimension,
  computeStrides
} from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { asStrided, asStrided_ } from "../creation";

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
