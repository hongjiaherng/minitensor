/**
 * shape & strides
 * - computeExpandedStrides (exposed to user's API)
 * - computeStrides (exposed to user's API)
 * - inferShape
 * - inferUnknownDimension
 * - areShapesEqual
 * - assertValidShape
 *
 */

import assert from "assert";
import { TensorLike, RecursiveArray } from "./types";
import { isTypedArray } from "./types_util";

export function computeStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

export function computeExpandedStrides(
  inputShape: number[],
  inputStrides: number[],
  targetShape: number[]
): number[] {
  const expandedStrides = new Array<number>(targetShape.length);
  for (let i = 0; i < targetShape.length; i++) {
    const reversedIndex = -(i + 1); // -1, -2, -3, ..., -targetShape.length
    if (inputShape.at(reversedIndex) === targetShape.at(reversedIndex)) {
      expandedStrides[targetShape.length - 1 - i] = inputStrides.at(
        reversedIndex
      ) as number;
    } else {
      // either undefined or not equal (broadcasted)
      expandedStrides[targetShape.length - 1 - i] = 0;
    }
  }
  return expandedStrides;
}

export function inferShape(
  data: TensorLike | RecursiveArray,
  shape: number[] = []
): number[] {
  if (!Array.isArray(data) && !isTypedArray(data)) {
    return [1];
  }
  shape.push(data.length);
  for (const element of data) {
    if (Array.isArray(element) || isTypedArray(element)) {
      inferShape(element, shape);
      break;
    }
  }
  return shape;
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

export function areShapesEqual(...shapes: number[][]): boolean {
  if (shapes.length < 2) return true;
  const firstShape = shapes[0];
  for (let i = 1; i < shapes.length; i++) {
    if (shapes[i].length !== firstShape.length) return false;
    for (let j = 0; j < firstShape.length; j++) {
      if (shapes[i][j] !== firstShape[j]) return false;
    }
  }
  return true;
}

export function assertValidShape(shape: number[]): void {
  assert(shape.length > 0, "Shape must be at least 1-dimensional");
  assert(
    shape.every((s) => s > 0),
    "Each dimension of shape must be positive value"
  );
}
