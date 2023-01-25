import { Tensor } from "../tensor";
import { DataType } from "../types";
import { computeStrides, isEqualShape } from "../utils";

export function reshape<D extends DataType>(
	x: Tensor<D>,
	shape: number[]
): Tensor<D> {
	if (isEqualShape(x.shape, shape)) {
		return x;
	}
	if (x.size !== shape.reduce((a, b) => a * b, 1)) {
		throw new Error(
			`Cannot reshape tensor of size ${x.size} to shape [${shape}]`
		);
	}
	const newX = new Tensor(x.data, x.shape, x.dtype);
	newX.shape = shape;
	newX.strides = computeStrides(shape);
	return newX;
}
