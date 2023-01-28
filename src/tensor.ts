import { Storage } from "./storage";
import { DType, RecursiveArray, TensorLike, TypedArrayMap } from "./types";
import { add } from "./ops/add";
import { mul } from "./ops/mul";
import { reshape } from "./ops/reshape";

export class Tensor<D extends DType> {
	data: Storage<D>;
	grad: Storage<D> | null;
	// and other autograd related properties

	constructor(data: TypedArrayMap[D], shape: number[], dtype?: D, broadcasted?: { oriShape: number[]; oriStrides: number[] }) {
		this.data = new Storage(data, shape, dtype, broadcasted);
		this.grad = null;
	}

	size(): number {
		return this.data.size;
	}

	shape(): number[] {
		return this.data.shape;
	}

	dtype(): D {
		return this.data.dtype;
	}

	view(): RecursiveArray {
		return this.data.view();
	}

	add(other: Tensor<D> | TensorLike): Tensor<D> {
		return add(this, other);
	}

	mul(other: Tensor<D> | TensorLike): Tensor<D> {
		return mul(this, other);
	}

	reshape(shape: number[]): Tensor<D> {
		return reshape(this, shape);
	}
}
