import { add } from "./ops";
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

	get(...indices: number[]): SingleValueMap[D] {
		const offset = this._indicesToOffset(indices);
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

	add(other: Tensor<D> | TensorLike): Tensor<D> {
		return add(this, other) as Tensor<D>;
	}

	_indicesToOffset(indices: number[]): number {
		if (indices.length !== this.shape.length) {
			throw new Error(
				`Tensor shape [${this.shape}] does not match indices length ${indices.length}`
			);
		}
		let offset = 0;
		for (let i = 0; i < indices.length; i++) {
			offset += indices[i] * this.strides[i];
		}
		return offset;
	}
}
