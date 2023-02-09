import { Tensor } from "../../tensor";
import { DType, PrimTypeMap } from "../../types";
import { reshape } from "../view/reshape";
import { tensor } from "./tensor";

export function full<D extends DType>(
	shape: number[],
	value: PrimTypeMap[D],
	dtype: D = "float32" as D
): Tensor<D> {
	const data = Array(shape.reduce((a, b) => a * b, 1)).fill(value);
	dtype = typeof value === "boolean" ? ("bool" as D) : dtype;
	return reshape(tensor(data, dtype), shape);
}
