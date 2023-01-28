import { Tensor } from "./tensor";
import {
	DType,
	RecursiveArray,
	TensorLike,
	TypedArray,
	TypedArrayMap,
} from "./types";
import {
	assertShape,
	castToTypedArray,
	computeBroadcastedStrides,
	computeStrides,
	emptyTypedArray,
	inferDTypeFromTypedArray,
	isEqualShape,
} from "./utils";

export class Storage<D extends DType> {
	dataInMemory: TypedArrayMap[D];
	size: number;
	shape: number[];
	strides: number[];
	dtype: D;
	_broadcasted: boolean = false;
	getByIndex: (index: number) => number;
	setByIndex: (value: number, index: number) => void;

	constructor(
		dataInMemory: TypedArrayMap[D],
		shape: number[],
		dtype?: D,
		broadcasted?: { oriShape: number[]; oriStrides: number[] }
	) {
		this.dataInMemory = dataInMemory;
		this.shape = shape;
		this.size = this.shape.reduce((a, b) => a * b, 1);
		this.dtype = dtype || inferDTypeFromTypedArray(dataInMemory);

		this.strides = !broadcasted
			? computeStrides(shape)
			: computeBroadcastedStrides(
					broadcasted.oriShape,
					broadcasted.oriStrides,
					shape
			  );

		// settings for broadcasted tensor
		this._broadcasted = !!broadcasted;
		if (this._broadcasted) {
			this.getByIndex = (index: number): number => {
				const offset = this._indexToOffset(index);
				return this.dataInMemory[offset];
			};
			this.setByIndex = (value: number, index: number): void => {
				const offset = this._indexToOffset(index);
				this.dataInMemory[offset] = value;
			};
		} else {
			this.getByIndex = (index: number): number => {
				return this.dataInMemory[index];
			};
			this.setByIndex = (value: number, index: number): void => {
				this.dataInMemory[index] = value;
			};
		}
	}

	get(loc: number[]): number {
		const offset = this._locToOffset(loc);
		return this.dataInMemory[offset];
	}

	set(value: number, loc: number[]) {
		const offset = this._locToOffset(loc);
		this.dataInMemory[offset] = value;
	}

	slice(begin: number[], size: number[]): Storage<D> {
		const sliced = castToTypedArray(
			this._sliceRecursive(begin, size),
			this.dtype
		);

		return new Storage(sliced, size, this.dtype);
	}

	setBySlice(values: Storage<D>, begin: number[], size: number[]) {
		if (values.size !== size.reduce((a, b) => a * b, 1)) {
			throw new Error("Size mismatch");
		}
		this._setBySliceRecursive(begin, size, values);
	}

	_setBySliceRecursive(begin: number[], size: number[], values: Storage<D>) {
		const _sliceRec = (
			begin: number[],
			axis: number,
			valuesInfo: { index: number }
		) => {
			if (axis === begin.length - 1) {
				const beginOffsetThis = this._locToOffset(begin);
				for (let i = 0; i < size[axis]; i++) {
					this.dataInMemory[beginOffsetThis + i] = values.getByIndex(
						valuesInfo.index
					);
					valuesInfo.index += 1;
				}
				return;
			}

			for (let i = 0; i < size[axis]; i++) {
				_sliceRec([...begin], axis + 1, valuesInfo);
				begin[axis] += 1;
			}
			return;
		};

		let valuesInfo = { index: 0 };
		_sliceRec(begin, 0, valuesInfo);
	}

	_sliceRecursive(begin: number[], size: number[]): number[] {
		const _sliceRec = (begin: number[], axis: number, array: number[]) => {
			if (axis === begin.length - 1) {
				const beginOffset = this._locToOffset(begin);
				array.push(
					...Array.from(
						this.dataInMemory.slice(beginOffset, beginOffset + size[axis])
					)
				);
				return;
			}

			for (let i = 0; i < size[axis]; i++) {
				_sliceRec([...begin], axis + 1, array);
				begin[axis] += 1;
			}
			return array;
		};

		const array = [] as number[];
		_sliceRec(begin, 0, array);
		return array;
	}

	view(): RecursiveArray {
		const _recursiveArray = (shape: number[], offset: number) => {
			if (shape.length === 1) {
				return Array.from(this.dataInMemory.slice(offset, offset + shape[0]));
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

	_indexToOffset(index: number): number {
		let offset = 0;
		for (let i = this.shape.length - 1; i >= 0; i--) {
			offset += (index % this.shape[i]) * this.strides[i];
			index = Math.floor(index / this.shape[i]);
		}
		return offset;
	}

	static emptyStorage<D extends DType>(shape: number[], dtype: D): Storage<D> {
		assertShape(shape);
		const dataInMemory = emptyTypedArray(
			shape.reduce((a, b) => a * b, 1),
			dtype
		);
		return new Storage(dataInMemory, shape, dtype);
	}
}
