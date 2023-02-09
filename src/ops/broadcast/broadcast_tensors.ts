import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { isEqualShape } from "../../utils";
import { broadcastShapes } from "./broadcast_shapes";
import { _broadcastTo } from "./broadcast_to";

export function broadcastTensors<D extends DType>(
	input: Tensor<D>,
	other: Tensor<D>
): [Tensor<D>, Tensor<D>] {
	// use broadcastShapes to get the new shape
	// use broadcastTo to get the new tensors (decide which one to broadcast)
	// return two tensors with the same shape (broadcasted)

	if (isEqualShape(input.shape, other.shape)) {
		return [input, other];
	}

	const targetShape = broadcastShapes(input.shape, other.shape);  // broacastable check is done here
	const broadcastedInput = _broadcastTo(input, targetShape);
	const broadcastedOther = _broadcastTo(other, targetShape);
	return [broadcastedInput, broadcastedOther];
}
