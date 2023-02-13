import { DType, PrimTypeMap, TensorLike, TypedArrayMap } from "./types";
import { inferDTypeFromTensorLikeObj, castToTypedArray } from "./types";

// Definition of "Storages are immutable":
// - size, dtype cannot be mutated
// - storage can be mutated in-place, but only by mutating the underlying TypedArray (not assigning a new TypedArray reference)


export class Storage<D extends DType> {
  public readonly size: number;
  public readonly dtype: D;
  public readonly storage: TypedArrayMap[D];

  constructor(data: TensorLike, dtype?: D) {
    this.dtype = dtype ?? inferDTypeFromTensorLikeObj(data) as D;
    this.storage = castToTypedArray(data, this.dtype);
    this.size = this.storage.length;
  }

  get(index: number): number {
    return this.storage[index];
  }

}
