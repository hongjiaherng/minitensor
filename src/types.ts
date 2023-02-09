import { isTypedArray } from "util/types";

export interface TypedArrayMap {
	float32: Float32Array;
	float64: Float64Array;
	int16: Int16Array;
	int32: Int32Array;
	bool: Uint8Array;
}

export interface PrimTypeMap {
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

// Order of precedence of type upcasting: float64 > float32 > int32 > int16 > bool

enum UpcastInt16AndMap {
	"float32" = "float32",
	"float64" = "float64",
	"int16" = "int16",
	"int32" = "int32",
	"bool" = "int16",
}

enum UpcastInt32AndMap {
	"float32" = "float32",
	"float64" = "float64",
	"int16" = "int32",
	"int32" = "int32",
	"bool" = "int32",
}

enum UpcastBoolAndMap {
	"float32" = "float32",
	"float64" = "float64",
	"int16" = "int16",
	"int32" = "int32",
	"bool" = "bool",
}

enum UpcastFloat64AndMap {
	"float32" = "float64",
	"float64" = "float64",
	"int16" = "float64",
	"int32" = "float64",
	"bool" = "float64",
}

enum UpcastFloat32AndMap {
	"float32" = "float32",
	"float64" = "float64",
	"int16" = "float32",
	"int32" = "float32",
	"bool" = "float32",
}

const upcastTypeMap = {
	float32: UpcastFloat32AndMap,
	float64: UpcastFloat64AndMap,
	int16: UpcastInt16AndMap,
	int32: UpcastInt32AndMap,
	bool: UpcastBoolAndMap,
};

export function upcastType(inputType: DType, otherType: DType): DType {
	return upcastTypeMap[inputType][otherType];
}

export function inferDTypeFromTensorLikeObj(tensorLikeObj: TensorLike): DType {
	if (
		typeof tensorLikeObj === "number" ||
		(Array.isArray(tensorLikeObj) && typeof tensorLikeObj[0] === "number")
	) {
		return "float32";
	} else if (
		typeof tensorLikeObj === "boolean" ||
		(Array.isArray(tensorLikeObj) && typeof tensorLikeObj[0] === "boolean")
	) {
		return "bool";
	} else if (isTypedArray(tensorLikeObj)) {
		return inferDTypeFromTypedArray(tensorLikeObj);
	}
	throw new Error("Cannot infer DType from TensorLike object");
}

export function inferDTypeFromTypedArray<D extends DType>(
	data: TypedArrayMap[D]
): D {
	if (data instanceof Float64Array) {
		return "float64" as D;
	} else if (data instanceof Float32Array) {
		return "float32" as D;
	} else if (data instanceof Int16Array) {
		return "int16" as D;
	} else if (data instanceof Int32Array) {
		return "int32" as D;
	} else if (data instanceof Uint8Array) {
		if (data.some((d) => d !== 0 && d !== 1)) {
			throw new Error(
				"Uint8Array should contain only 0s and 1s to be casted to bool"
			);
		}
		return "bool" as D;
	}
	throw new Error("Cannot infer DType from TypedArray object");
}

export function castToTypedArray<D extends DType>(
	tensorLikeObj: TensorLike,
	dtype: D
): TypedArrayMap[D] {
	let typedArray: TypedArray;

	if (
		isTypedArray(tensorLikeObj) &&
		isDTypeMatchedWithTypedArray(tensorLikeObj, dtype)
	) {
		return tensorLikeObj as TypedArrayMap[D];
	}

	if (!Array.isArray(tensorLikeObj) && !isTypedArray(tensorLikeObj)) {
		tensorLikeObj = [tensorLikeObj] as number[] | boolean[];
	}

	switch (dtype) {
		case "float32":
			typedArray = Float32Array.from(tensorLikeObj as Array<number>);
			break;
		case "float64":
			typedArray = Float64Array.from(tensorLikeObj as Array<number>);
			break;
		case "int16":
			typedArray = Int16Array.from(tensorLikeObj as Array<number>);
			break;
		case "int32":
			typedArray = Int32Array.from(tensorLikeObj as Array<number>);
			break;
		case "bool":
			if (
				Array.isArray(tensorLikeObj) &&
				typeof tensorLikeObj[0] === "boolean"
			) {
				typedArray = Uint8Array.from(tensorLikeObj as Array<number>);
			} else {
				if (
					(tensorLikeObj as TypedArray | Array<number>).some(
						(d) => d !== 0 && d !== 1
					)
				) {
					throw new Error(
						"tensorLikeObj should contain only 0s and 1s to be casted to bool"
					);
				}
				typedArray = Uint8Array.from(tensorLikeObj as Array<number>);
			}
			break;

		default:
			throw new Error("Invalid dtype");
	}

	return typedArray as TypedArrayMap[D];
}

export function isDTypeMatchedWithTypedArray<D extends DType>(
	typedArray: TypedArray,
	dtype: D
): boolean {
	if (!isTypedArray(typedArray)) return false;
	try {
		return inferDTypeFromTypedArray(typedArray) === dtype;
	} catch (error) {
		return false;
	}
}
