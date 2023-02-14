import { areShapesEqual } from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { DType, PrimTypeMap } from "../../types";
import { expand } from "../view";
import { broadcastShapes } from "./broadcast_shapes";

export function isBroadcastedTensor<D extends DType>(
  tensor: Tensor<D>
): number {
  return Number(tensor.strides.some((stride) => stride === 0));
}

// tensor iterators
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
          t.size !== tensors[0].size ||
          !areShapesEqual(t.shape, tensors[0].shape)
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
      value: value
    };
  }

  [Symbol.iterator](): TensorsIterator<D> {
    return this;
  }
}

export function areBroadcastableTogether<D extends DType>(
  ...tensors: Tensor<D>[]
): boolean {
  if (tensors.length < 2) throw new Error("At least 2 tensors are required");
  try {
    broadcastShapes(...tensors.map((t) => t.shape));
    return true;
  } catch (error) {
    return false;
  }
}

export function isBroadcastableTo<D extends DType>(
  tensor: Tensor<D>,
  shape: number[]
): boolean {
  try {
    expand(tensor, shape);
    return true;
  } catch (error) {
    return false;
  }
}
