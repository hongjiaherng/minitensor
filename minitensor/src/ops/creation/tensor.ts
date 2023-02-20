import assert from "assert";
import {
  inferShape,
  computeStrides,
  assertValidShape
} from "../../shape_strides_util";
import { flattenArray, Storage } from "../../storage";
import { Tensor } from "../../tensor";
import { DType, RecursiveArray, TensorLike } from "../../types";

export function tensor<D extends DType>(
  data: TensorLike | RecursiveArray,
  shape: number[] = inferShape(data),
  dtype?: D
): Tensor<D> {
  const flattenedData = flattenArray(data);

  assertValidShape(shape);
  assert(
    flattenedData.length === shape.reduce((a, b) => a * b, 1),
    `Shape [${shape}] does not match data length ${flattenedData.length}`
  );

  const strides = computeStrides(shape);
  const offset = 0;
  const storage = new Storage(flattenedData as TensorLike, dtype);
  return new Tensor(storage, shape, strides, offset);
}
