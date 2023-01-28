import { Tensor } from "../tensor";
import { DType, TensorLike } from "../types";
import { convertToTensor, determineDType } from "../utils";
import { ensureShapesMatch } from "./broadcast";
import { empty } from "./tensor_init";

export function mul<D extends DType>(
	a: Tensor<D> | TensorLike,
	b: Tensor<D> | TensorLike
): Tensor<D> {
	let result: Tensor<D>;
	let a_ = a instanceof Tensor ? a : convertToTensor(a);
	let b_ = b instanceof Tensor ? b : convertToTensor(b);
	
	// ensure that the shapes are equal (broadcasting done here)
	[a_, b_] = ensureShapesMatch(a_, b_);

	result = empty(
		a_.shape(),
		determineDType(a_.dtype(), b_.dtype())
	) as Tensor<D>;

	for (let i = 0; i < result.size(); i++) {
		// console.log(a_.data.getByIndex(i), b_.data.getByIndex(i));
		result.data.setByIndex(a_.data.getByIndex(i) * b_.data.getByIndex(i), i);
	}

	return result;
}
