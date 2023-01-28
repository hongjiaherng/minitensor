import { Tensor } from "../tensor";
import { DType, DTypeMap } from "../types";
import { isEqualShape } from "../utils";

export function broadcastShapes(shapeA: number[], shapeB: number[]): number[] {
	// broadcastable check is done here
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

export function broadcastTensors<D extends DType>(
	a: Tensor<D>,
	b: Tensor<D>
): [Tensor<D>, Tensor<D>] {
	// use broadcastShapes to get the new shape
	// use broadcastTo to get the new tensors (decide which one to broadcast)
	// return two tensors with the same shape (broadcasted)
	const targetShape = broadcastShapes(a.data.shape, b.data.shape);
	const broadcastedA = _broadcastTo(a, targetShape);
	const broadcastedB = _broadcastTo(b, targetShape);
	return [broadcastedA, broadcastedB];
}

export function ensureShapesMatch<D extends DType>(a: Tensor<D>, b: Tensor<D>) {
	if (!isEqualShape(a.shape(), b.shape())) {
		[a, b] = broadcastTensors(a, b);
	}
	return [a, b];
}

export function _broadcastTo<D extends DType>(x: Tensor<D>, shape: number[]) {
	// no validation here, assume the shape is valid
	if (isEqualShape(x.shape(), shape)) return x;
	const newX = new Tensor(x.data.dataInMemory, shape, x.data.dtype, {
		oriShape: x.data.shape,
		oriStrides: x.data.strides,
	});
	return newX;
}

export class TensorsIterator<D extends DType>
	implements Iterator<DTypeMap[D][]>
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
					t.size !== tensors[0].size ||
					!isEqualShape(t.shape(), tensors[0].shape())
			)
		) {
			throw new Error("All tensors must have the same shape");
		}
		this.tensors = tensors;
		this.done = false;
		this.index = 0;
	}

	next(): IteratorResult<DTypeMap[D][]> {
		if (this.done) {
			return { done: this.done, value: undefined };
		}
		if (this.index === this.tensors[0].size()) {
			// all tensros have the same size / shape, checking tensors[0] is enough
			this.done = true;
			return { done: this.done, value: undefined };
		}

		const values = this.tensors.map((t) => t.data._indexToOffset(this.index));

		this.index += 1;
		return {
			done: false,
			value: values as DTypeMap[D][],
		};
	}

	[Symbol.iterator](): TensorsIterator<D> {
		return this;
	}
}
