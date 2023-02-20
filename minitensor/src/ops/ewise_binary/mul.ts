import { Tensor } from "../../tensor";
import { DType, TensorLike, RecursiveArray, PrimTypeMap } from "../../types";
import { upcastType } from "../../types_util";
import { broadcastTensors, broadcastTo, TensorsIterator } from "../broadcast";
import { empty, tensor } from "../creation";

export function mul<D1 extends DType, D2 extends DType>(
  input: Tensor<D1> | TensorLike | RecursiveArray,
  other: Tensor<D2> | TensorLike | RecursiveArray
) {
  let input_ = input instanceof Tensor ? input : tensor(input);
  let other_ = other instanceof Tensor ? other : tensor(other);

  [input_, other_] = broadcastTensors(input_, other_);

  // const targetArray = [];
  const targetDType = upcastType(input_.dtype, other_.dtype);
  const resultedTensor = empty(input_.shape, targetDType);

  const tensorsIterator = new TensorsIterator(input_, other_);

  if (targetDType !== DType.bool) {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      inputVal = Number(inputVal);
      otherVal = Number(otherVal);
      resultedTensor._setByIndex(i, (inputVal * otherVal) as any);
    });
  } else {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      resultedTensor._setByIndex(i, inputVal && otherVal);
    });
  }

  return resultedTensor;
}

export function mul_<D1 extends DType, D2 extends DType>(
  input: Tensor<D1>,
  other: Tensor<D2> | TensorLike | RecursiveArray
) {
  let other_ = other instanceof Tensor ? other : (tensor(other) as Tensor<D2>);
  other_ = broadcastTo(other_, input.shape);
  const otherSameTypeAsInput = other_.type(input.dtype);
  const tensorsIterator = new TensorsIterator(input, otherSameTypeAsInput);

  if (input.dtype !== DType.bool) {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      inputVal = Number(inputVal);
      otherVal = Number(otherVal);
      input._setByIndex(i, (inputVal * otherVal) as any);
    });
  } else {
    tensorsIterator.forEach(([inputVal, otherVal], i) => {
      input._setByIndex(i, inputVal && otherVal);
    });
  }

  return input;
}
