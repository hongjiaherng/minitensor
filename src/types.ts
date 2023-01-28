export interface TypedArrayMap {
	float32: Float32Array;
	float64: Float64Array;
	int16: Int16Array;
	int32: Int32Array;
	bool: Uint8Array;
}

export interface DTypeMap {
	float32: number;
	float64: number;
	int16: number;
	int32: number;
	bool: boolean;
}

export type DType = keyof TypedArrayMap;
export type TypedArray = TypedArrayMap[DType];
export type TensorLike = number | boolean | number[] | boolean[] | TypedArray;
export type RecursiveArray = Array<RecursiveArray | number | boolean>;
