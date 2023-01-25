import { Tensor } from "../tensor";
import { tensor } from "../tensor_init";
import { DataType, SingleValueMap } from "../types";
import { computeStrides } from "../utils";

export function broadcastTo(x: Tensor<DataType>, shape: number[]) {
	if (
		x.shape.length === shape.length &&
		x.shape.every((v, i) => v === shape[i])
	) {
		// if tensor is already the same shape as shape, return tensor
		return x;
	}
	const newX = tensor(x.data, x.shape, x.dtype);
	newX.shape = shape;
	newX.size = shape.reduce((a, b) => a * b, 1);
	newX.strides = computeBroadcastedStrides(x.shape, x.strides, shape);

	return newX;
}

export function computeBroadcastedStrides(
	oriShape: number[],
	oriStrides: number[],
	newShape: number[]
) {
	const oriShape_ = [...oriShape]; // make deepcopy
	const oriStrides_ = [...oriStrides];

	while (oriShape_.length != newShape.length) {
		if (oriShape_.length < newShape.length) {
			oriShape_.unshift(1);
			oriStrides_.unshift(0);
		}
	}

	const newStrides = [];
	for (let i = oriShape_.length - 1; i >= 0; i--) {
		if (oriShape_[i] === newShape[i]) {
			newStrides.unshift(oriStrides_[i]);
		} else {
			newStrides.unshift(0);
		}
	}
	return newStrides;
}

export function broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
	const newShape: number[] = [];
	const shapeA_ = [...shapeA]; // make deepcopy
	const shapeB_ = [...shapeB];

	while (shapeA_.length != shapeB_.length) {
		if (shapeA_.length > shapeB_.length) {
			shapeB_.unshift(1);
		} else {
			shapeA_.unshift(1);
		}
	}

	for (let i = shapeA_.length - 1; i >= 0; i--) {
		if (shapeA_[i] !== shapeB_[i] && shapeA_[i] !== 1 && shapeB_[i] !== 1) {
			throw new Error(
				`Operands could not be broadcast together with shapes [${shapeA}] and [${shapeB}]`
			);
		}
		newShape.unshift(Math.max(shapeA_[i], shapeB_[i]));
	}
	return newShape;
}

export function broadcastTensors(
	tensorA: Tensor<DataType>,
	tensorB: Tensor<DataType>
) {
	// use broadcastShapes to get the new shape
	// use broadcastTo to get the new tensors (decide which one to broadcast)
	// return two tensors with the same shape (broadcasted)
	const targetShape = broadcastShapes(tensorA.shape, tensorB.shape);
	const broadcastedTensorA = broadcastTo(tensorA, targetShape);
	const broadcastedTensorB = broadcastTo(tensorB, targetShape);
	return [broadcastedTensorA, broadcastedTensorB];
}

// export class BroadcastedTensorsIterator<D extends DataType>
// 	implements Iterator<D>
// {
// 	private tensorA: Tensor<D>;
// 	private tensorB: Tensor<D>;
// 	private done: boolean;

// 	constructor(tensorA: Tensor<D>, tensorB: Tensor<D>) {
// 		this.tensorA = tensorA;
// 		this.tensorB = tensorB;
// 		this.done = false;
// 	}

// 	public next(): IteratorResult<D, number[] | undefined> {
// 		if (this.done) {
// 			return { done: this.done, value: undefined };
// 		}

// 	}

// 	public [Symbol.iterator](): BroadcastedTensorsIterator<D> {
// 		return this;
// 	}
// }
