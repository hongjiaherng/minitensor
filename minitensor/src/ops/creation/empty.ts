import { assertValidShape } from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { DType, TypedArrayMap } from "../../types";
import { tensor } from "./tensor";

export function empty<D extends DType>(
  shape: number[],
  dtype: D = DType.float32 as D
): Tensor<D> {
  assertValidShape(shape);

  const data = createEmptyTypedArray(
    dtype,
    shape.reduce((a, b) => a * b, 1)
  );
  return tensor(data, shape, dtype) as Tensor<D>;
}

function createEmptyTypedArray<D extends DType>(
  dtype: D,
  size: number
): TypedArrayMap[D] {
  switch (dtype) {
    case DType.float32:
      return new Float32Array(size) as TypedArrayMap[D];
    case DType.float64:
      return new Float64Array(size) as TypedArrayMap[D];
    case DType.int16:
      return new Int16Array(size) as TypedArrayMap[D];
    case DType.int32:
      return new Int32Array(size) as TypedArrayMap[D];
    case DType.bool:
      return new Uint8Array(size) as TypedArrayMap[D];
    default:
      throw new Error("Invalid dtype");
  }
}
