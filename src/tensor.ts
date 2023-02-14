import { isBroadcastedTensor, TensorsIterator } from "./ops/broadcast";
import { tensor } from "./ops/creation";
import { expand } from "./ops/view";
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
  public getByIndex: (index: number) => PrimTypeMap[D];
  public setByIndex: (index: number, value: PrimTypeMap[D]) => void;

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

    this.getByIndex = createGetByIndexMethod(this);
    this.setByIndex = createSetByIndexMethod(this);
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

  public set(data: Tensor<D> | TensorLike | RecursiveArray): void {
    // TODO: add support for whatever dtype of Tensor
    /**
     * cannot be used on broadcasted tensors, this must be a non-broadcasted tensor
     */
    // handle scalar value or tensor with size 1
    // Broadcast input to this.shape
    // set data by iterating through
    if (isBroadcastedTensor(this))
      throw new Error(
        "unsupported operation: cannot set values on broadcasted / expanded tensors. Please clone() the tensor before performing the operation."
      );

    const subTensor =
      data instanceof Tensor ? data : tensor(data, undefined, this.dtype);
    const subTensor_ = expand(subTensor, this.shape); // broadcast to this.shape if necessary
    const tensorsIterator = new TensorsIterator(subTensor_);

    tensorsIterator.forEach(([subTensorVal], i) => {
      this.setByIndex(i, subTensorVal);
    });
  }

  _indexToOffset(index: number): number {
    let offset = this.offset;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      offset += (index % this.shape[i]) * this.strides[i];
      index = Math.floor(index / this.shape[i]);
    }
    return offset;
  }
}

// factory method to create a getByIndexMethod based on strides and dtype
function createGetByIndexMethod<D extends DType>(tensor: Tensor<D>) {
  let getByIndex: (index: number) => PrimTypeMap[D];
  if (isBroadcastedTensor(tensor)) {
    if (tensor.dtype === "bool") {
      getByIndex = (index: number) =>
        !!tensor.data.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
    } else {
      getByIndex = (index: number) =>
        tensor.data.get(tensor._indexToOffset(index)) as PrimTypeMap[D];
    }
  } else {
    if (tensor.dtype === "bool") {
      getByIndex = (index: number) =>
        !!tensor.data.get(index) as PrimTypeMap[D];
    } else {
      getByIndex = (index: number) => tensor.data.get(index) as PrimTypeMap[D];
    }
  }
  return getByIndex;
}

function createSetByIndexMethod<D extends DType>(tensor: Tensor<D>) {
  let setByIndex: (index: number, value: PrimTypeMap[D]) => void;
  if (isBroadcastedTensor(tensor)) {
    setByIndex = (index: number, value: PrimTypeMap[D]) =>
      tensor.data.set(tensor._indexToOffset(index), Number(value));
  } else {
    setByIndex = (index: number, value: PrimTypeMap[D]) =>
      tensor.data.set(index, Number(value));
  }
  return setByIndex;
}
