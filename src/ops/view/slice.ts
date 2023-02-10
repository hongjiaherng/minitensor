import { Tensor } from "../../tensor";
import { DType } from "../../types";

export function slice<D extends DType>(
	x: Tensor<D>,
	begin: number[],
	size: number[]
): Tensor<D> {
	// resulted: shape, strides, offset
	const shape = size;
	const strides = x.strides.map(
		(stride, i) => stride * (size[i] === 1 ? 0 : 1)
	);
	const offset =
		x.offset + x.strides.reduce((acc, stride, i) => acc + stride * begin[i], 0);
	return new Tensor<D>(x.data, shape, strides, offset);
}

/**
 *
 * input =
 * [[1, 2, 3],
 * [4, 5, 6],
 * [7, 8, 9]]
 *
 * test 1:
 * input[0] or input [0, :]
 * begin = [0, 0]
 * size = [1, 3]
 *
 * test 2:
 * input[0:1, :]
 * begin = [0, 0]
 * size = [1, 3]
 *
 * test 3:
 * input[0:2, :]
 * begin = [0, 0]
 * size = [2, 3]
 *
 * test 4:
 * input[0:2, 0:2]
 * begin = [0, 0]
 * size = [2, 2]
 *
 * test 5:
 * input[1, 1]
 * begin = [1, 1]
 * size = [1, 1]
 */
