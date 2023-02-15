export enum DType {
  float64 = "float64",
  float32 = "float32",
  int32 = "int32",
  int16 = "int16",
  bool = "bool"
}

export interface TypedArrayMap {
  [DType.float32]: Float32Array;
  [DType.float64]: Float64Array;
  [DType.int16]: Int16Array;
  [DType.int32]: Int32Array;
  [DType.bool]: Uint8Array;
}

export interface PrimTypeMap {
  [DType.float32]: number;
  [DType.float64]: number;
  [DType.int16]: number;
  [DType.int32]: number;
  [DType.bool]: boolean;
}

export type TypedArray = TypedArrayMap[DType];
export type TensorLike = number | boolean | number[] | boolean[] | TypedArray;
export type RecursiveArray = Array<RecursiveArray | number | boolean>;
