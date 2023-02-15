import { Tensor } from "../../tensor";
import { DType, TensorLike, RecursiveArray } from "../../types";
import { upcastType } from "../../types_util";
import { broadcastTensors, TensorsIterator } from "../broadcast";
import { empty, tensor } from "../creation";

export function add<D1 extends DType, D2 extends DType>(
  input: Tensor<D1> | TensorLike | RecursiveArray,
  other: Tensor<D2> | TensorLike | RecursiveArray
) {
  // convert to tensor if not already
  // check if shapes are compatible and broadcast if not
  // determine target dtype
  // create new tensor with target dtype

  let input_ = input instanceof Tensor ? input : tensor(input);
  let other_ = other instanceof Tensor ? other : tensor(other);

  [input_, other_] = broadcastTensors(input_, other_);

  // const targetArray = [];
  const targetDType = upcastType(input_.dtype, other_.dtype);
  const resultedTensor = empty(input_.shape, targetDType);

  const tensorsIterator = new TensorsIterator(input_, other_);

  if (targetDType === DType.bool) {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      resultedTensor._setByIndex(i, inputVal || otherVal);
    });
  } else {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      inputVal = Number(inputVal);
      otherVal = Number(otherVal);
      resultedTensor._setByIndex(i, inputVal + otherVal);
    });
  }

  return resultedTensor;
}
