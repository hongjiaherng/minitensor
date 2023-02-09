import { Storage } from "./storage";
import { DType, PrimTypeMap, RecursiveArray, TensorLike } from "./types";
import {
	isBroadcasted,
} from "./utils";

export class Tensor<D extends DType> {
	public readonly shape: number[];
	public readonly strides: number[];
	public readonly offset: number;
	public readonly storage: Storage<D>;
	public readonly dtype: D;
	public readonly size: number;
	public getByIndex: (index: number) => PrimTypeMap[D];

	constructor(
		storage: Storage<D>,
		shape: number[],
		strides: number[],
		offset: number
	) {
		// TODO: add validation for shape, strides, offset
		this.shape = shape;
		this.strides = strides;
		this.offset = offset;
		this.storage = storage;
		this.dtype = this.storage.dtype;
		this.size = this.shape.reduce((a, b) => a * b, 1);
		this.getByIndex = createGetByIndexMethod(this);
	}

	public data(): RecursiveArray {
		const _recursiveArray = (shape: number[], offset: number) => {
			if (shape.length === 1) {
				let slicedArray: number[] | boolean[] = Array.from(
					this.storage.storage.slice(offset, offset + shape[0])
				);
				if (this.dtype === "bool") {
					slicedArray = slicedArray.map((v) => !!v);
				}
				return slicedArray;
			}
			let size = shape.shift() ?? 0;
			let array = new Array(size);
			for (let i = 0; i < size; i++) {
				array[i] = _recursiveArray(
					[...shape],
					i * this.strides[this.strides.length - shape.length - 1] + offset
				);
			}
			return array;
		};

		return _recursiveArray([...this.shape], 0);
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

function createGetByIndexMethod<D extends DType>(tensor: Tensor<D>) {
	let getByIndex: (index: number) => PrimTypeMap[D];
	if (isBroadcasted(tensor)) {
		if (tensor.dtype === "bool") {
			getByIndex = (index: number) =>
				!!tensor.storage.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
		} else {
			getByIndex = (index: number) =>
				tensor.storage.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
		}
	} else {
		if (tensor.dtype === "bool") {
			getByIndex = (index: number) =>
				!!tensor.storage.get(index) as PrimTypeMap[D];
		} else {
			getByIndex = (index: number) =>
				tensor.storage.get(index) as PrimTypeMap[D];
		}
	}
	return getByIndex;
}
