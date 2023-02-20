import { isBroadcastedTensor, TensorsIterator } from "./ops/broadcast";
import { tensor } from "./ops/creation";
import { add, add_, mul, mul_ } from "./ops/ewise_binary";
import {
  expand,
  reshape,
  reshape_,
  slice,
  squeeze,
  squeeze_
} from "./ops/view";
import { Selection } from "./ops/view/slice";
import { Storage } from "./storage";
import { DType, PrimTypeMap, RecursiveArray, TensorLike } from "./types";

/**
 * Tensors are immutable
 *
 * Definition of "Tensors are immutable":
 * - shape, strides, offset, size can be mutated
 * - reference to data cannot be mutated, but only by mutating the underlying data's storage (not assigning a new storage reference)
 * - dtype cannot be mutated
 *
 */

export class Tensor<D extends DType> {
  public shape: number[];
  public strides: number[];
  public offset: number;
  public readonly data: Storage<D>;
  public readonly dtype: D;
  public size: number;

  constructor(
    data: Storage<D>,
    shape: number[],
    strides: number[],
    offset: number
  ) {
    // TODO: add validation for shape, strides, offset
    this.shape = shape;
    this.strides = strides;
    this.offset = offset;
    this.data = data;
    this.dtype = this.data.dtype;
    this.size = this.shape.reduce((a, b) => a * b, 1);
  }

  public array(): RecursiveArray {
    const _recursiveArray = (shape: number[], offset: number) => {
      if (shape.length === 1) {
        const stride = this.strides[this.strides.length - 1]; // stride of innermost dimension
        const slicedArray = Array.from(
          { length: shape[0] },
          (_, i) => this.data.storage[offset + i * stride]
        );
        return this.dtype === "bool"
          ? slicedArray.map((v) => !!v)
          : slicedArray;
      }
      let size = shape.shift()!;
      let array = new Array(size);
      for (let i = 0; i < size; i++) {
        array[i] = _recursiveArray(
          [...shape],
          i * this.strides[this.strides.length - shape.length - 1] + offset
        );
      }
      return array;
    };

    return _recursiveArray([...this.shape], this.offset);
  }

  public set<T extends DType>(
    data: Tensor<T> | TensorLike | RecursiveArray
  ): void {
    /**
     * cannot be used on broadcasted tensors, this must be a non-broadcasted tensor
     */
    if (isBroadcastedTensor(this))
      throw new Error(
        "unsupported operation: cannot set values on broadcasted / expanded tensors. Please clone() the tensor before performing the operation."
      );

    let subTensor: Tensor<D> =
      data instanceof Tensor
        ? data.type(this.dtype)
        : tensor(data, undefined, this.dtype);

    subTensor = expand(subTensor, this.shape); // broadcast to this.shape if necessary
    const tensorsIterator = new TensorsIterator(subTensor);

    tensorsIterator.forEach(([subTensorVal], i) => {
      this._setByIndex(i, subTensorVal);
    });
  }

  public clone(): Tensor<D> {
    // check if this is a broadcasted tensor
    if (isBroadcastedTensor(this)) {
      // create tensor using the nested array as it will produce the actual mutidimensional array without broadcasting
      return tensor(this.array(), [...this.shape], this.dtype);
    }
    return new Tensor(this.data.clone(), [...this.shape], [...this.strides], 0);
  }

  public type<T extends DType>(dtype: T): Tensor<T> {
    if ((this.dtype as DType) === (dtype as DType))
      return this as unknown as Tensor<T>; // no need to convert
    if (isBroadcastedTensor(this)) {
      return tensor(this.array(), [...this.shape], dtype);
    }
    return new Tensor(
      this.data.type(dtype),
      [...this.shape],
      [...this.strides],
      0
    );
  }

  public _getByIndex(index: number): PrimTypeMap[D] {
    return this.dtype !== DType.bool
      ? (this.data.get(this._indexToOffset(index)) as PrimTypeMap[D])
      : (!!this.data.get(this._indexToOffset(index)) as PrimTypeMap[D]);
  }

  public _setByIndex(index: number, value: PrimTypeMap[D]): void {
    this.data.set(this._indexToOffset(index), Number(value));
  }

  _indexToOffset(index: number): number {
    let offset = this.offset;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      offset += (index % this.shape[i]) * this.strides[i];
      index = Math.floor(index / this.shape[i]);
    }
    return offset;
  }

  add<T extends DType>(
    other: Tensor<T> | TensorLike | RecursiveArray
  ): Tensor<T | D> {
    return add(this, other) as Tensor<T | D>;
  }

  add_<T extends DType>(
    other: Tensor<T> | TensorLike | RecursiveArray
  ): Tensor<D> {
    return add_(this, other);
  }

  mul<T extends DType>(
    other: Tensor<T> | TensorLike | RecursiveArray
  ): Tensor<T | D> {
    return mul(this, other) as Tensor<T | D>;
  }

  mul_<T extends DType>(
    other: Tensor<T> | TensorLike | RecursiveArray
  ): Tensor<D> {
    return mul_(this, other);
  }

  reshape(shape: number[]): Tensor<D> {
    return reshape(this, shape);
  }

  reshape_(shape: number[]): Tensor<D> {
    return reshape_(this, shape);
  }

  expand(shape: number[]): Tensor<D> {
    return expand(this, shape);
  }

  slice(selection: Selection, keepDim: boolean = false): Tensor<D> {
    return slice(this, selection, keepDim);
  }

  squeeze(dim?: number | number[]): Tensor<D> {
    return squeeze(this, dim);
  }

  squeeze_(dim?: number | number[]): Tensor<D> {
    return squeeze_(this, dim);
  }

  broadcastTo(shape: number[]): Tensor<D> {
    return this.expand(shape);
  }
}
