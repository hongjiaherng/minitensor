import { Tensor } from "../../tensor";
import { DType, TensorLike, RecursiveArray } from "../../types";
import { tensor } from "../creation";

export function add<D extends DType>(
  input: Tensor<D> | TensorLike | RecursiveArray,
  other: Tensor<D> | TensorLike | RecursiveArray
): Tensor<D> {
  // convert to tensor if not already
  // check if shapes are compatible and broadcast if not
  // determine target dtype
  // create new tensor with target dtype

  // let input_ = input instanceof Tensor ? input : tensor(input);
  // let other_ = other instanceof Tensor ? other : tensor(other);

  // [input_, other_] = broadcastTensors(input_, other_);

  // const targetArray = [];
  // const targetDType = upcastType(input_.dtype, other_.dtype);

  // const tensorsIterator = new TensorsIterator(input_, other_);

  // if (targetDType === "bool") {
  //   for (const [inputVal, otherVal] of tensorsIterator) {
  //     targetArray.push((inputVal as boolean) || (otherVal as boolean));
  //   }
  // } else {
  //   for (const [inputVal, otherVal] of tensorsIterator) {
  //     targetArray.push(Number(inputVal) + Number(otherVal));
  //   }
  // }

  // return fromTensorLike(
  //   targetArray as TensorLike,
  //   input_.shape,
  //   targetDType
  // ) as Tensor<D>;
  return input instanceof Tensor ? input : tensor(input);
}
