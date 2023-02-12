import { Storage } from "./storage";
import { DType, PrimTypeMap, RecursiveArray, TensorLike } from "./types";
import { isBroadcasted } from "./utils";

/**
 * Tensors are immutable
 *
 * Definition of "Tensors are immutable":
 * - shape, strides, offset, dtype, size cannot be mutated
 * - data can be mutated, but only by mutating the underlying storage's storage (not assigning a new storage reference)
 *
 */

export class Tensor<D extends DType> {
	public readonly shape: number[];
	public readonly strides: number[];
	public readonly offset: number;
	public readonly data: Storage<D>;
	public readonly dtype: D;
	public readonly size: number;
	public getByIndex: (index: number) => PrimTypeMap[D];

	constructor(
		data: Storage<D>,
		shape: number[],
		strides: number[],
		offset: number
	) {
		// TODO: add validation for shape, strides, offset
		this.shape = shape;
		this.strides = strides;
		this.offset = offset;
		this.data = data;
		this.dtype = this.data.dtype;
		this.size = this.shape.reduce((a, b) => a * b, 1);

		this.getByIndex = createGetByIndexMethod(this);
	}

	public array(): RecursiveArray {
		const _recursiveArray = (shape: number[], offset: number) => {
			if (shape.length === 1) {
				const stride = this.strides[this.strides.length - 1]; // stride of innermost dimension
				const slicedArray = Array.from(
					{ length: shape[0] },
					(_, i) => this.data.storage[offset + i * stride]
				);
				return this.dtype === "bool"
					? slicedArray.map((v) => !!v)
					: slicedArray;
			}
			let size = shape.shift()!;
			let array = new Array(size);
			for (let i = 0; i < size; i++) {
				array[i] = _recursiveArray(
					[...shape],
					i * this.strides[this.strides.length - shape.length - 1] + offset
				);
			}
			return array;
		};

		return _recursiveArray([...this.shape], this.offset);
	}

	_indexToOffset(index: number): number {
		let offset = 0;
		for (let i = this.shape.length - 1; i >= 0; i--) {
			offset += (index % this.shape[i]) * this.strides[i];
			index = Math.floor(index / this.shape[i]);
		}
		return offset;
	}
}

// factory method to create a getByIndexMethod based on strides and dtype
function createGetByIndexMethod<D extends DType>(tensor: Tensor<D>) {
	let getByIndex: (index: number) => PrimTypeMap[D];
	if (isBroadcasted(tensor)) {
		if (tensor.dtype === "bool") {
			getByIndex = (index: number) =>
				!!tensor.data.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
		} else {
			getByIndex = (index: number) =>
				tensor.data.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
		}
	} else {
		if (tensor.dtype === "bool") {
			getByIndex = (index: number) =>
				!!tensor.data.get(index) as PrimTypeMap[D];
		} else {
			getByIndex = (index: number) => tensor.data.get(index) as PrimTypeMap[D];
		}
	}
	return getByIndex;
}
