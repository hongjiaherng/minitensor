import { inferShape, computeStrides } from "../../shape_strides_util";
import { flattenArray, Storage } from "../../storage";
import { Tensor } from "../../tensor";
import { DType, RecursiveArray, TensorLike } from "../../types";

export function tensor<D extends DType>(
  data: TensorLike | RecursiveArray,
  shape: number[] = inferShape(data),
  dtype?: D,
): Tensor<D> {
  const strides = computeStrides(shape);
  const offset = 0;
  const storage = new Storage(flattenArray(data) as TensorLike, dtype);
  return new Tensor(storage, shape, strides, offset);
}
