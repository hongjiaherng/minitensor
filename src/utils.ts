import { DataType, ArrayLike, TypedArray } from "./types";

export function computeStrides(shape: number[]): number[] {
	const strides = new Array(shape.length);
	let stride = 1;
	for (let i = shape.length - 1; i >= 0; i--) {
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
}

export function inferShape(data: ArrayLike): number[] {
	return [data.length];
}

export function inferDType(data: ArrayLike): DataType {
	if (data instanceof Float32Array) return "float32";
	else if (data instanceof Int32Array || data instanceof Uint8Array)
		return "int32";
	else if (typeof data[0] === "number") return "float32";
	else if (typeof data[0] === "boolean") return "bool";
	return "float32";
}

export function castToDType(data: ArrayLike, dtype: DataType): TypedArray {
	let newData: TypedArray;
	switch (dtype) {
		case "float32":
			if (Array.isArray(data) && typeof data[0] !== "number") {
				throw new Error("Data doesn't match the type of the given dtype");
			}
			newData = Float32Array.from(data as number[] | TypedArray);
			break;

		case "int32":
			if (Array.isArray(data) && typeof data[0] !== "number") {
				throw new Error("Data doesn't match the type of the given dtype");
			}
			newData = Int32Array.from(data as number[] | TypedArray);
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
				newData = data;
			} else {
				throw new Error("Data doesn't match the type of the given dtype");
			}
			break;
		default:
			throw Error("Invalid dtype");
	}
	return newData;
}
