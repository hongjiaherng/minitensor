import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { asStrided } from "../creation";
import {
  areShapesEqual,
  assertValidShape,
  computeExpandedStrides
} from "../../shape_strides_util";
import { broadcastShapes } from "../broadcast";

export function expand<D extends DType>(
  input: Tensor<D>,
  shape: number[]
): Tensor<D> {
  assertValidShape(shape); // validate shape
  
  if (shape.length < input.shape.length) {
    throw new Error(
      `Incompatible shape: the rank of shape provided (${shape.length}) must be greater or equal to the rank of the input tensor (${input.shape.length})`
    );
  }

  if (areShapesEqual(input.shape, shape)) return input;

  try {
    const broadcastedShape = broadcastShapes(input.shape, shape); // validate shapes, might throw unbroadcastable error
    if (!areShapesEqual(shape, broadcastedShape)) throw new Error(); // input shape & target shape are broadcastable, but the broadcasted shape is not equal to the target shape
  } catch (error) {
    throw new Error(
      `Incompatible shape: the tensor of shape [${input.shape}] cannot be expanded to the shape provided [${shape}]`
    );
  }

  const expandedStrides = computeExpandedStrides(
    input.shape,
    input.strides,
    shape
  );
  return asStrided(input, shape, expandedStrides, input.offset);
}
