import {
  areShapesEqual,
  computeExpandedStrides
} from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { asStrided } from "../creation";
import { expand } from "../view";

export function broadcastTo<D extends DType>(
  input: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // exposed to user, validate shape
  // call view.expand internally
  return expand(input, shape);
}

export function _broadcastTo<D extends DType>(
  input: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // no validation here, assume the shape is valid
  // check if shape is compatible with input shape
  // create new tensor with same storage, modify shape, strides, and offset (no change)
  if (areShapesEqual(input.shape, shape)) return input;

  const expandedStrides = computeExpandedStrides(
    input.shape,
    input.strides,
    shape
  );
  return asStrided(input, shape, expandedStrides, input.offset);
}
