export interface DataTypeMap {
  float32: Float32Array;
  int32: Int32Array;
  bool: Uint8Array;
}

export interface SingleValueMap {
  float32: number;
  int32: number;
  bool: boolean;
}

export type RecursiveArray = Array<RecursiveArray | number | boolean>;
export type DataType = keyof DataTypeMap;
export type TypedArray = Float32Array | Uint8Array | Int32Array;
export type TensorLike = number | boolean | number[] | boolean[] | TypedArray;