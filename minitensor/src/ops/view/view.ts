import {
  computeStrides,
  inferUnknownDimension
} from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { isContiguous } from "../../tensor_util";
import { DType } from "../../types";
import { asStrided } from "../creation";

export function view<D extends DType>(
  x: Tensor<D>,
  shape: number[]
): Tensor<D> {
  if (isContiguous(x)) {
    shape = inferUnknownDimension(x.size, shape);
    return asStrided(x, shape, computeStrides(shape));
  } else {
    throw new Error(
      "Cannot create a view of a non-contiguous tensor. Use reshape() instead."
    );
  }
}
