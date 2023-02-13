import { Tensor } from "../../tensor";
import { DType, PrimTypeMap } from "../../types";
import { isEqualShape } from "../../utils";

export class TensorsIterator<D extends DType>
  implements Iterator<PrimTypeMap[D][]>
{
  private tensors: Tensor<D>[];
  private done: boolean;
  private index: number;

  constructor(...tensors: Tensor<D>[]) {
    if (tensors.length === 0)
      throw new Error("At least one tensor is required");

    if (
      tensors.some(
        (t) =>
          t.size !== tensors[0].size || !isEqualShape(t.shape, tensors[0].shape)
      )
    )
      throw new Error("All tensors must have the same shape and size");

    this.tensors = tensors;
    this.done = false;
    this.index = 0;
  }

  next(): IteratorResult<PrimTypeMap[D][]> {
    if (this.done) return { done: true, value: undefined };
    if (this.index >= this.tensors[0].size) {
      this.done = true;
      return { done: true, value: undefined };
    }
    const value = this.tensors.map((t) => t.getByIndex(this.index));
    this.index += 1;
    return {
      done: false,
      value: value,
    };
  }

  [Symbol.iterator](): TensorsIterator<D> {
    return this;
  }
}
