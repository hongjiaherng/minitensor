export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}

export type DataType = keyof DataTypeMap;
export type TypedArray = Float32Array | Uint8Array | Int32Array;
export type ArrayLike = number[] | boolean[] | TypedArray;