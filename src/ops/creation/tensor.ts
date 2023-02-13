import { Storage } from "../../storage";
import { Tensor } from "../../tensor";
import { DType, RecursiveArray, TensorLike } from "../../types";
import { computeStrides, flattenArray, inferShape } from "../../utils";

export function tensor<D extends DType>(
  data: TensorLike | RecursiveArray,
  dtype?: D
): Tensor<D> {
  const shape = inferShape(data);
  const strides = computeStrides(shape);
  const offset = 0;
  const storage = new Storage(flattenArray(data) as TensorLike, dtype);
  return new Tensor(storage, shape, strides, offset);
}
