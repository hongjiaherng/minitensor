import { DType, PrimTypeMap, TensorLike, TypedArrayMap } from "./types";
import { inferDTypeFromTensorLikeObj, castToTypedArray } from "./types";

export class Storage<D extends DType> {
  size: number;
  dtype: D;
  storage: TypedArrayMap[D];

  constructor(data: TensorLike, dtype?: D) {
    this.dtype = dtype ?? inferDTypeFromTensorLikeObj(data) as D;
    this.storage = castToTypedArray(data, this.dtype);
    this.size = this.storage.length;
  }

  get(index: number): number {
    return this.storage[index];
  }

}
