import { Tensor } from "../../tensor";
import { DType } from "../../types";

export function asStrided<D extends DType>(
	x: Tensor<D>,
	shape: number[],
	strides: number[],
	offset: number = 0
): Tensor<D> {
	// create new tensor with same storage, modify shape, strides, and offset (no change)
	return new Tensor(x.data, shape, strides, offset);
}
