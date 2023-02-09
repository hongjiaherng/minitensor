import assert from "assert";
import { isTypedArray } from "util/types";
import { Tensor } from "./tensor";
import {
	DType,
	inferDTypeFromTypedArray,
	RecursiveArray,
	TensorLike,
	TypedArray,
	TypedArrayMap,
} from "./types";

export function createEmptyTypedArray<D extends DType>(size: number, dtype: D) {
	switch (dtype) {
		case "float32":
			return new Float32Array(size);
		case "float64":
			return new Float64Array(size);
		case "int16":
			return new Int16Array(size);
		case "int32":
			return new Int32Array(size);
		case "bool":
			return new Uint8Array(size);
		default:
			throw new Error("Unsupported dtype");
	}
}

export function computeStrides(shape: number[]): number[] {
	const strides = new Array(shape.length);
	let stride = 1;
	for (let i = shape.length - 1; i >= 0; i--) {
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
}

export function inferShape(
	data: TensorLike | RecursiveArray,
	shape: number[] = []
): number[] {
	if (!Array.isArray(data) && !isTypedArray(data)) {
		return [1];
	}
	shape.push(data.length);
	for (const element of data) {
		if (Array.isArray(element) || isTypedArray(element)) {
			inferShape(element, shape);
			break;
		}
	}
	return shape;
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

export function assertValidShape(shape: number[]): void {
	assert(shape.length > 0, "Shape must be at least 1-dimensional");
	assert(
		shape.every((s) => s > 0),
		"Each dimension of shape must be positive value"
	);
}

// export function assertShape(shape: number[], size?: number): void {
// 	assert(shape.length > 0, "Shape must be at least 1-dimensional");
// 	assert(
// 		shape.every((s) => s > 0),
// 		"Each dimension of shape must be positive value"
// 	);
// 	if (size) {
// 		assert(
// 			shape.reduce((a, b) => a * b) === size,
// 			`Shape [${shape}] does not compatible with size ${size}`
// 		);
// 	}
// }

export function isEqualShape(
	inputShape: number[],
	otherShape: number[]
): boolean {
	if (inputShape.length !== otherShape.length) return false;
	for (let i = 0; i < inputShape.length; i++) {
		if (inputShape[i] !== otherShape[i]) return false;
	}
	return true;
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

export function isBroadcasted<D extends DType>(tensor: Tensor<D>): number {
	return Number(tensor.strides.some((stride) => stride === 0));
}

// export function emptyTypedArray<D extends DType>(
// 	size: number,
// 	dtype: D
// ): TypedArrayMap[D] {
// 	let typedArray;
// 	switch (dtype) {
// 		case "float32":
// 			typedArray = new Float32Array(size);
// 			break;
// 		case "float64":
// 			typedArray = new Float64Array(size);
// 			break;
// 		case "int16":
// 			typedArray = new Int16Array(size);
// 			break;
// 		case "int32":
// 			typedArray = new Int32Array(size);
// 			break;
// 		case "bool":
// 			typedArray = new Uint8Array(size);
// 			break;
// 		default:
// 			throw new Error("Invalid data type");
// 	}
// 	return typedArray as TypedArrayMap[D];
// }
