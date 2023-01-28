import { Tensor } from "../tensor";
import { DType, TensorLike, TypedArrayMap } from "../types";
import {
	assertShape,
	castToTypedArray,
	emptyTypedArray,
	inferDTypeFromTensorLikeObj,
	inferShape,
} from "../utils";

export function tensor<D extends DType>(
	data: TensorLike,
	shape?: number[],
	dtype?: D
): Tensor<D> {
	let newData: TypedArrayMap[D];

	// verify dtype
	if (!dtype) {
		dtype = inferDTypeFromTensorLikeObj(data) as D;
	}
	newData = castToTypedArray(data, dtype);

	// verify shape
	if (!shape) {
		shape = inferShape(newData);
	}
	if (shape) assertShape(shape, newData.length);

	return new Tensor(newData, shape, dtype);
}

export function empty<D extends DType>(
	shape: number[],
	dtype: D = "float32" as D
): Tensor<D> {
	assertShape(shape);
	const size = shape.reduce((a, b) => a * b, 1);
	let data: TypedArrayMap[D];
	switch (dtype) {
		case "float32":
			data = new Float32Array(size) as TypedArrayMap[D];
			break;
		case "float64":
			data = new Float64Array(size) as TypedArrayMap[D];
			break;
		case "int16":
			data = new Int16Array(size) as TypedArrayMap[D];
			break;
		case "int32":
			data = new Int32Array(size) as TypedArrayMap[D];
			break;
		case "bool":
			data = new Uint8Array(size) as TypedArrayMap[D];
			break;
		default:
			throw new Error(`Unknown dtype: ${dtype}`);
	}
	return new Tensor(data, shape, dtype);
}

export function arange<D extends DType>(
	start: number,
	end: number,
	step: number = 1,
	dtype: D = "float32" as D
): Tensor<D> {
	const size = Math.ceil((end - start) / step);
	let data: TypedArrayMap[D];
	switch (dtype) {
		case "float64":
			data = new Float64Array(size) as TypedArrayMap[D];
			break;
		case "float32":
			data = new Float32Array(size) as TypedArrayMap[D];
			break;
		case "int32":
			data = new Int32Array(size) as TypedArrayMap[D];
			break;
		case "int16":
			data = new Int16Array(size) as TypedArrayMap[D];
			break;
		case "bool":
			throw new Error("Cannot create bool array from arange");
		default:
			throw new Error(`Unknown dtype: ${dtype}`);
	}
	for (let i = 0; i < size; i++) {
		data[i] = start + i * step;
	}
	return new Tensor(data, [size], dtype);
}

export function full<D extends DType>(
	shape: number[],
	value: number,
	dtype: D = "float32" as D
) {
	assertShape(shape);
	const size = shape.reduce((a, b) => a * b, 1);
	const data = emptyTypedArray(size, dtype).fill(value);
	return new Tensor(data as TypedArrayMap[D], shape, dtype);
}
