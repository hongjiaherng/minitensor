import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { computeBroadcastedStrides, isEqualShape } from "../../utils";
import { asStrided } from "../creation/as_strided";
import { isBroadcastable } from "./is_broadcastable";

export function _broadcastTo<D extends DType>(
  input: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // no validation here, assume the shape is valid
  // check if shape is compatible with input shape
  // create new tensor with same storage, modify shape, strides, and offset (no change)

  if (isEqualShape(input.shape, shape)) return input;

  const newStrides = computeBroadcastedStrides(
    input.shape,
    input.strides,
    shape
  );

  return asStrided(input, shape, newStrides, input.offset);
}

export function broadcastTo<D extends DType>(
  input: Tensor<D>,
  shape: number[]
): Tensor<D> {
  // this function is not used internally, it is exposed for user to check if two shapes are broadcastable, inefficient operations due to alot of checks

  if (!isBroadcastable(input.shape, shape))
    throw new Error(
      `Incompatible shapes to be braodcasted together: [${input.shape}] and [${shape}]`
    );
  return _broadcastTo(input, shape);
}
