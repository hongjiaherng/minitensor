import { isTypedArray } from "util/types";
import { Tensor } from "./tensor";
import { DType, RecursiveArray, TensorLike, TypedArrayMap } from "./types";
import { castToTypedArray, inferDTypeFromTensorLikeObj } from "./types_util";

// Definition of "Storages are immutable":
// - size, dtype cannot be mutated
// - storage can be mutated in-place, but only by mutating the underlying TypedArray (not assigning a new TypedArray reference)

export class Storage<D extends DType> {
  public readonly size: number;
  public readonly dtype: D;
  public readonly storage: TypedArrayMap[D];

  constructor(data: TensorLike, dtype?: D) {
    this.dtype = dtype ?? (inferDTypeFromTensorLikeObj(data) as D);
    this.storage = castToTypedArray(data, this.dtype);
    this.size = this.storage.length;
  }

  get(index: number): number {
    return this.storage[index];
  }
}

export function flattenArray(
  data: TensorLike | RecursiveArray
): (number | boolean)[] {
  if (!Array.isArray(data) && !isTypedArray(data)) {
    return [data];
  }
  const flatArray = [];
  for (let i = 0; i < data.length; i++) {
    flatArray.push(...flattenArray(data[i]));
  }
  return flatArray;
}

export function sharesMemory<D extends DType>(
  x: Tensor<D>,
  y: Tensor<D>
): boolean {
  return x.data === y.data;
}
