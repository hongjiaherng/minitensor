import { Storage } from "../../storage";
import { Tensor } from "../../tensor";
import { DType, RecursiveArray, TensorLike } from "../../types";
import { computeStrides, flattenArray, inferShape } from "../../utils";

export function fromTensorLike<D extends DType>(
	data: TensorLike,
	shape: number[] = inferShape(data),
	dtype?: D
): Tensor<D> {
	const strides = computeStrides(shape);
	const offset = 0;
	const storage = new Storage(data, dtype);
	return new Tensor(storage, shape, strides, offset);
}
