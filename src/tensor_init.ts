import { isTypedArray } from "util/types";
import { Tensor } from "./tensor";
import { ArrayLike, DataType, TypedArray } from "./types";
import { inferShape, inferDType, castToDType } from "./utils";

export function tensor(
	data: ArrayLike,
	shape?: number[],
	dtype?: DataType
): Tensor {
	// verify data
	if (!isTypedArray(data) && !Array.isArray(data)) {
		throw new Error(
			"Data must be either a TypedArray or an Array of numbers or booleans"
		);
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
	data = castToDType(data, dtype); // cast data to dtype

	return new Tensor(data, shape, dtype);
}
