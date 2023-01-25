import { add, reshape } from "./ops";
import {
	DataType,
	DataTypeMap,
	RecursiveArray,
	SingleValueMap,
	TensorLike,
} from "./types";
import { castArrayToDType, computeStrides } from "./utils";

export class Tensor<D extends DataType> {
	shape: number[];
	size: number;
	data: DataTypeMap[D];
	strides: number[];
	dtype: D;

	constructor(data: DataTypeMap[D], shape: number[], dtype?: D) {
		this.shape = shape;
		this.size = this.shape.reduce((a, b) => a * b, 1);
		this.data = dtype
			? data
			: (castArrayToDType(data, "float32") as DataTypeMap[D]);
		this.dtype = dtype || ("float32" as D);
		this.strides = computeStrides(shape);
	}

	get(...loc: number[]): SingleValueMap[D] {
		const offset = this._locToOffset(loc);
		return this.data[offset] as SingleValueMap[D];
	}

	view(): RecursiveArray {
		const _recursiveArray = (shape: number[], offset: number) => {
			if (shape.length === 1) {
				// console.log(offset, "->", offset + shape[0]);
				// console.log(this.data.slice(offset, offset + shape[0]));
				return Array.from(this.data.slice(offset, offset + shape[0]));
			}
			let size = shape.shift() ?? 0;
			let array = new Array(size);
			for (let i = 0; i < size; i++) {
				// console.log(i * this.strides[this.strides.length - shape.length - 1] + offset)
				array[i] = _recursiveArray(
					[...shape],
					i * this.strides[this.strides.length - shape.length - 1] + offset
				);
			}
			return array;
		};

		return _recursiveArray([...this.shape], 0);
	}

	reshape(shape: number[]): Tensor<D> {
		return reshape(this, shape);
	}

	add(other: Tensor<D> | TensorLike): Tensor<D> {
		return add(this, other);
	}

	_locToOffset(loc: number[]): number {
		if (loc.length !== this.shape.length) {
			throw new Error(
				`Tensor shape [${this.shape}] does not match loc length ${loc.length}`
			);
		}
		let offset = 0;
		for (let i = 0; i < loc.length; i++) {
			offset += loc[i] * this.strides[i];
		}
		return offset;
	}

	_indexToLoc(index: number): number[] {
		const loc = new Array(this.shape.length);
		for (let i = this.shape.length - 1; i >= 0; i--) {
			loc[i] = index % this.shape[i];
			index = Math.floor(index / this.shape[i]);
		}
		return loc;
	}
}
