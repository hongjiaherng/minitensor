import { Tensor } from "./tensor";
import { DType, RecursiveArray, TensorLike, TypedArrayMap } from "./types";
import { castToTypedArray, inferDTypeFromTensorLikeObj, isTypedArray } from "./types_util";

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

  public get(index: number): number {
    return this.storage[index];
  }

  public set(index: number, value: number): void {
    this.storage[index] = value;
  }

  public clone(): Storage<D> {
    return new Storage(this.storage.slice(), this.dtype);
  }

  public type<T extends DType>(dtype: T): Storage<T> {
    if ((this.dtype as DType) === (dtype as DType)) {
      // create new storage with same data, but different dtype since Storage is immutable
      return this as unknown as Storage<T>;
    }
    return new Storage(this.storage, dtype);
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
