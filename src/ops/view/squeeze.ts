import { DType } from "../../types";
import { Tensor } from "../../tensor";
import { asStrided, asStrided_ } from "../creation";

export function squeeze<D extends DType>(
  input: Tensor<D>,
  dim: number | number[] = []
): Tensor<D> {
  const { squeezedShape, squeezedStrides } = getSqueezedShapeAndStrides(
    input.shape,
    input.strides,
    dim
  );
  return asStrided(input, squeezedShape, squeezedStrides);
}

export function squeeze_<D extends DType>(
  input: Tensor<D>,
  dim: number | number[] = []
): Tensor<D> {
  // In-place version of squeeze
  const { squeezedShape, squeezedStrides } = getSqueezedShapeAndStrides(
    input.shape,
    input.strides,
    dim
  );
  return asStrided_(input, squeezedShape, squeezedStrides);
}

function getSqueezedShapeAndStrides(
  shape: number[],
  strides: number[],
  dim: number | number[] = []
): { squeezedShape: number[]; squeezedStrides: number[] } {
  if (typeof dim === "number") dim = [dim];
  const squeezedShape = [] as number[];
  const squeezedStrides = [] as number[];

  for (let i = 0; i < shape.length; i++) {
    // If the dimension is not 1, or if the dimension is 1 but we want to keep it
    if (shape[i] !== 1 || (dim.length !== 0 && !dim.includes(i))) {
      squeezedShape.push(shape[i]);
      squeezedStrides.push(strides[i]);
    }
  }
  if (squeezedShape.length === 0) {
    squeezedShape.push(1);
    squeezedStrides.push(1);
  }
  return { squeezedShape, squeezedStrides };
}
