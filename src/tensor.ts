import { DataType, TypedArray } from "./types";
import { computeStrides } from "./utils";

export class Tensor {
	shape: number[];
	size: number;
	data: TypedArray;
	strides: number[];
	dtype: DataType;

	constructor(data: TypedArray, shape: number[], dtype?: DataType) {
		this.shape = shape;
		this.size = this.shape.reduce((a, b) => a * b, 1);
		this.data = data;
		this.dtype = dtype || ("float32" as DataType);
		this.strides = computeStrides(shape);
	}
}
