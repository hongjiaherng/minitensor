import assert from "assert";
import { isTypedArray } from "util/types";
import { tensor } from "./ops/tensor_init";
import { Tensor } from "./tensor";
import { DType, TensorLike, TypedArray, TypedArrayMap } from "./types";

export function computeStrides(shape: number[]): number[] {
	const strides = new Array(shape.length);
	let stride = 1;
	for (let i = shape.length - 1; i >= 0; i--) {
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
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
		return "bool" as D;
	} else {
		throw new Error("Invalid data type");
	}
}

export function inferDTypeFromTensorLikeObj(data: TensorLike): DType {
	if (
		typeof data === "number" ||
		(Array.isArray(data) && typeof data[0] === "number")
	) {
		return "float32";
	} else if (
		typeof data === "boolean" ||
		(Array.isArray(data) && typeof data[0] === "boolean")
	) {
		return "bool";
	} else if (isTypedArray(data)) {
		return inferDTypeFromTypedArray(data);
	} else {
		throw new Error("Invalid data type");
	}
}

export function castToTypedArray<D extends DType>(
	data: TensorLike,
	dtype: D
): TypedArrayMap[D] {
	let newData: TypedArray;

	if (isTypedArray(data) && inferDTypeFromTypedArray(data) === dtype) {
		return data as TypedArrayMap[D];
	}

	if (!Array.isArray(data) && !isTypedArray(data)) {
		data = [data] as number[] | boolean[];
	}

	switch (dtype) {
		case "float32":
			newData = Float32Array.from(data as number[] | TypedArrayMap[D]);
			break;
		case "float64":
			newData = Float64Array.from(data as number[] | TypedArrayMap[D]);
			break;

		case "int16":
			newData = Int16Array.from(data as number[] | TypedArrayMap[D]);
			break;

		case "int32":
			newData = Int32Array.from(data as number[] | TypedArrayMap[D]);
			break;

		case "bool":
			// if bool[] -> Uint8Array
			// if Uint8Array with 0 and 1 only -> do nothing
			// otherwise throw error
			if (Array.isArray(data) && typeof data[0] === "boolean") {
				newData = new Uint8Array(data.length);
				data.forEach((d, i) => {
					if (d === true) newData[i] = 1;
				});
			} else if (data instanceof Uint8Array) {
				if (data.some((d) => d !== 0 && d !== 1)) {
					throw new Error(
						"Uint8Array should contain only 0s and 1s to be casted to bool"
					);
				}
				newData = data as TypedArrayMap[D];
			} else {
				throw new Error("Data is not castable to the given dtype");
			}
			break;
		default:
			throw new Error("Invalid data type");
	}
	return newData as TypedArrayMap[D];
}

export function inferShape(data: TensorLike): number[] {
	if (!Array.isArray(data) && !isTypedArray(data)) {
		return [1];
	}
	return [data.length];
}

export function assertShape(shape: number[], size?: number): void {
	assert(shape.length > 0, "Shape must be at least 1-dimensional");
	assert(
		shape.every((s) => s > 0),
		"Each dimension of shape must be positive value"
	);
	if (size) {
		assert(
			shape.reduce((a, b) => a * b) === size,
			`Shape [${shape}] does not compatible with size ${size}`
		);
	}
}

export function convertToTensor<D extends DType>(
	tensorLike: TensorLike
): Tensor<D> {
	return tensor(tensorLike);
}

export function isEqualShape(shapeA: number[], shapeB: number[]): boolean {
	if (shapeA.length !== shapeB.length) return false;
	for (let i = 0; i < shapeA.length; i++) {
		if (shapeA[i] !== shapeB[i]) return false;
	}
	return true;
}

export function determineDType(...dtypes: DType[]): DType {
	if (dtypes.length === 1) return dtypes[0];
	else if (dtypes.some((dtype) => dtype === "float64")) return "float64";
	else if (dtypes.some((dtype) => dtype === "float32")) return "float32";
	else if (dtypes.some((dtype) => dtype === "int32")) return "int32";
	else if (dtypes.some((dtype) => dtype === "int16")) return "int16";
	else return "bool";
}

export function computeBroadcastedStrides(
	oriShape: number[],
	oriStrides: number[],
	newShape: number[]
) {
	const oriShape_ = [...oriShape]; // make deepcopy
	const oriStrides_ = [...oriStrides];

	while (oriShape_.length != newShape.length) {
		if (oriShape_.length < newShape.length) {
			oriShape_.unshift(1);
			oriStrides_.unshift(0);
		}
	}

	const newStrides = [];
	for (let i = oriShape_.length - 1; i >= 0; i--) {
		if (oriShape_[i] === newShape[i]) {
			newStrides.unshift(oriStrides_[i]);
		} else {
			newStrides.unshift(0);
		}
	}

	return newStrides;
}

export function emptyTypedArray<D extends DType>(
	size: number,
	dtype: D
): TypedArrayMap[D] {
	let typedArray;
	switch (dtype) {
		case "float32":
			typedArray = new Float32Array(size);
			break;
		case "float64":
			typedArray = new Float64Array(size);
			break;
		case "int16":
			typedArray = new Int16Array(size);
			break;
		case "int32":
			typedArray = new Int32Array(size);
			break;
		case "bool":
			typedArray = new Uint8Array(size);
			break;
		default:
			throw new Error("Invalid data type");
	}
	return typedArray as TypedArrayMap[D];
}
