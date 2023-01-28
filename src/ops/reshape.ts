import { Tensor } from "../tensor";
import { DType } from "../types";
import { computeStrides, isEqualShape } from "../utils";
import { tensor } from "./tensor_init";

// TODO: check if this is correct

export function reshape<D extends DType>(
	x: Tensor<D>,
	shape: number[]
): Tensor<D> {
	if (isEqualShape(x.shape(), shape)) {
		return x;
	}
	if (x.size() !== shape.reduce((a, b) => a * b, 1)) {
		throw new Error(
			`Cannot reshape tensor of size ${x.size} to shape [${shape}]`
		);
	}
	const newX = new Tensor(x.data.dataInMemory, shape, x.dtype());
	return newX;
}
