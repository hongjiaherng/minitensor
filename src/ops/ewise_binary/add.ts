import { Tensor } from "../../tensor";
import { DType, RecursiveArray, TensorLike, upcastType } from "../../types";
import { broadcastTensors } from "../broadcast/broadcast_tensors";
import { TensorsIterator } from "../broadcast/tensors_iterator";
import { tensor } from "../creation/tensor";

export function add<D extends DType>(
	input: Tensor<D> | TensorLike | RecursiveArray,
	other: Tensor<D> | TensorLike | RecursiveArray
): Tensor<D> {
	// convert to tensor if not already
	// check if shapes are compatible and broadcast if not
	// determine target dtype
	// create new tensor with target dtype

	let input_ = input instanceof Tensor ? input : tensor(input);
	let other_ = other instanceof Tensor ? other : tensor(other);

	[input_, other_] = broadcastTensors(input_, other_);

  const targetArray = [];
	const targetDtype = upcastType(input_.dtype, other_.dtype);
	
  // TODO: settle boolean dtype
  // a = torch.tensor([True, False, True])
  // b = torch.tensor([False, False, True])
  // a + b
  // tensor([ True, False,  True])

  for (const [inputVal, otherVal] of new TensorsIterator(input_, other_)) {
    targetArray.push(inputVal + otherVal);
  }

  return input_ as Tensor<D>;

}
