import { assertValidShape } from "../../utils";

export function isBroadcastable(
  inputShape: number[],
  otherShape: number[]
): boolean {
  // this function is not used internally, it is exposed for user to check if two shapes are broadcastable, inefficient operations due to alot of checks
  assertValidShape(inputShape);
  assertValidShape(otherShape);

  const inputShape_ = [...inputShape]; // make deepcopy
  const otherShape_ = [...otherShape];

  // pad with 1s if shapes are not the same length
  while (inputShape_.length != otherShape_.length) {
    if (inputShape_.length < otherShape_.length) {
      inputShape_.unshift(1);
    } else {
      otherShape_.unshift(1);
    }
  }

  for (let i = inputShape_.length - 1; i >= 0; i--) {
    if (
      inputShape_[i] !== otherShape_[i] &&
      inputShape_[i] !== 1 &&
      otherShape_[i] !== 1
    ) {
      return false;
    }
  }

  return true;
}