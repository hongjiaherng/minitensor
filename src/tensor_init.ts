import { isTypedArray } from "util/types";
import { Tensor } from "./tensor";
import { DataType, TensorLike, TypedArray } from "./types";
import { inferShape, inferDType, castArrayToDType } from "./utils";

export function tensor(
	data: TensorLike,
	shape?: number[],
	dtype?: DataType
): Tensor<DataType> {
	// cast data to array if not already
	if (typeof data === "number" || typeof data === "boolean") {
		data = [data] as number[] | boolean[];
	}

	// verify shape
	if (shape != null && shape.length === 0)
		throw new Error("Shape must have at least one dimension");
	if (shape == null) shape = inferShape(data); // infer shape if null

	// verify size of data with specified shape
	const size = shape.reduce((a, b) => a * b, 1);
	if (data.length !== size) {
		throw new Error(
			`Tensor shape [${shape}] (size=${size}) does not match data length ${data.length}`
		);
	}

	if (dtype == null) dtype = inferDType(data); // infer dtype if null
	data = castArrayToDType(data, dtype); // cast data to dtype

	return new Tensor(data, shape, dtype);
}

export function empty(
	shape: number[],
	dtype: DataType = "float32"
): Tensor<DataType> {
	const size = shape.reduce((a, b) => a * b, 1);
	let data: TypedArray;
	switch (dtype) {
		case "float32":
			data = new Float32Array(size);
			break;
		case "int32":
			data = new Int32Array(size);
			break;
		case "bool":
			data = new Uint8Array(size);
			break;
		default:
			throw new Error(`Unknown dtype: ${dtype}`);
	}
	return new Tensor(data, shape, dtype);
}

export function arange(
	start: number,
	end: number,
	step: number = 1,
	dtype: DataType = "float32"
): Tensor<DataType> {
	const size = Math.ceil((end - start) / step);
	let data: TypedArray;
	switch (dtype) {
		case "float32":
			data = new Float32Array(size);
			break;
		case "int32":
			data = new Int32Array(size);
			break;
		case "bool":
			data = new Uint8Array(size);
			break;
		default:
			throw new Error(`Unknown dtype: ${dtype}`);
	}
	for (let i = 0; i < size; i++) {
		data[i] = start + i * step;
	}
	return new Tensor(data, [size], dtype);
}
