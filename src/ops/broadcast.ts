import { Tensor } from "../tensor";
import { DataType, DataTypeMap, SingleValueMap } from "../types";
import { computeStrides, isEqualShape } from "../utils";

export function broadcastTo(x: Tensor<DataType>, shape: number[]) {
	if (
		x.shape.length === shape.length &&
		x.shape.every((v, i) => v === shape[i])
	) {
		// if tensor is already the same shape as shape, return tensor
		return x;
	}
	const newX = new Tensor(x.data, x.shape, x.dtype);
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

export class TensorsIterator<D extends DataType>
	implements Iterator<SingleValueMap[D][]>
{
	private tensors: Tensor<D>[];
	private done: boolean;
	private index: number;

	constructor(...tensors: Tensor<D>[]) {
		if (tensors.length === 0) {
			throw new Error("No tensors provided");
		}
		if (
			tensors.some(
				(t) =>
					t.size !== tensors[0].size || !isEqualShape(t.shape, tensors[0].shape)
			)
		) {
			throw new Error("All tensors must have the same shape");
		}
		this.tensors = tensors;
		this.done = false;
		this.index = 0;
	}

	next(): IteratorResult<SingleValueMap[D][]> {
		if (this.done) {
			return { done: this.done, value: undefined };
		}
		if (this.index === this.tensors[0].size) {
			// all tensros have the same size / shape, checking tensors[0] is enough
			this.done = true;
			return { done: this.done, value: undefined };
		}

		const values = this.tensors.map(
			(t) => t.data[t._locToOffset(t._indexToLoc(this.index))]
		);
		this.index += 1;
		return {
			done: false,
			value: values as SingleValueMap[D][],
		};
	}

	[Symbol.iterator](): TensorsIterator<D> {
		return this;
	}
}
