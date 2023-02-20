import assert from "assert";
import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { asStrided } from "../creation";
import { squeeze_ } from "./squeeze";

interface SlicingSelectionPerDim {
  start?: number | null;
  stop?: number | null;
  step?: number;
}

type SelectionPerDim = SlicingSelectionPerDim | number | null;
export type Selection = SelectionPerDim[];

export function slice<D extends DType>(
  x: Tensor<D>,
  selection: Selection,
  keepDim: boolean = false
): Tensor<D> {
  /**
   * 1) selection -> beginIndices, slicedShape, slicedStrides
   *
   * selection[i] is either a number, a slicing selection, null, or undefined
   * - number: select a single entry
   * - slicing selection: select a range of entries
   * - null or undefined: select all entries
   *
   * 2) beginIndices -> slicedOffset
   *
   * 3) slicedStrides, slicedOffset, slicedShape -> new tensor view
   */

  // Validate selection
  assertSelectionIsValid(selection, x.shape);

  // Compute beginIndices, slicedShape, slicedStrides, and slicedOffset
  const { slicedShape, slicedStrides, slicedOffset } = computeSlicedTensorInfo(
    x,
    selection
  );
  
  // Create new tensor view with new strides, offset, and shape
  return keepDim
    ? asStrided(x, slicedShape, slicedStrides, slicedOffset)
    : squeeze_(asStrided(x, slicedShape, slicedStrides, slicedOffset));
}

function computeSlicedTensorInfo<D extends DType>(
  x: Tensor<D>,
  selection: Selection
): {
  slicedShape: number[];
  slicedStrides: number[];
  slicedOffset: number;
} {
  const beginIndices = new Array(x.shape.length);
  const slicedShape = new Array(x.shape.length);
  const slicedStrides = new Array(x.shape.length);
  let slicedOffset = x.offset;

  for (let i = 0; i < x.shape.length; i++) {
    if (selection[i] == null) {
      // null or undefined
      beginIndices[i] = 0;
      slicedShape[i] = x.shape[i];
      slicedStrides[i] = x.strides[i];
    } else if (typeof selection[i] === "number") {
      beginIndices[i] = handlePotentialNegativeIndex(
        selection[i] as number,
        x.shape[i]
      );
      slicedShape[i] = 1;
      slicedStrides[i] = x.strides[i];
    } else {
      const { start, stop, step } = handleSlicingSelectionPerDim(
        selection[i] as SlicingSelectionPerDim,
        x.shape[i]
      );
      beginIndices[i] = step > 0 ? start : stop - 1;
      slicedShape[i] = Math.ceil((stop - start) / Math.abs(step));
      slicedStrides[i] = x.strides[i] * step;
    }
    slicedOffset += beginIndices[i] * x.strides[i];
  }
  return { slicedShape, slicedStrides, slicedOffset };
}

function assertSelectionIsValid(selection: Selection, shape: number[]): void {
  // What to check?
  // - # of dimensions indexed is valid
  // - each dimension's selection is valid

  assert(
    selection.length <= shape.length && selection.length > 0,
    `Invalid number of indices for tensor: tensor is ${shape.length}-dimensional, but ${selection.length} dimensions were indexed`
  );
  selection.forEach((dimSelection, i) => {
    if (typeof dimSelection === "number") {
      // SingleIndexPerDim
      const start = handlePotentialNegativeIndex(dimSelection, shape[i]);
      assert(
        start >= 0 && start < shape[i],
        `Index ${dimSelection} is out of bounds for dimension with size ${shape[i]}`
      );
    } else if (dimSelection !== null && typeof dimSelection === "object") {
      // SlicingSelectionPerDim
      const { start, stop, step } = handleSlicingSelectionPerDim(
        dimSelection,
        shape[i]
      );
      assert(
        start < stop,
        `Invalid slice selection: start index ${start} must be smaller than end index ${stop}`
      );
      assert(step !== 0, "Step must be non-zero");
      assert(
        start >= 0 && stop <= shape[i],
        `Slice selection [${start}:${stop}:${step}] is out of bounds for dimension with size ${shape[i]}`
      );
    }
  });
}

function handleSlicingSelectionPerDim(
  dimSelection: SlicingSelectionPerDim,
  dimSize: number
) {
  let { start, stop, step } = dimSelection;
  start = start ? handlePotentialNegativeIndex(start, dimSize) : 0;
  stop = stop ? handlePotentialNegativeIndex(stop, dimSize) : dimSize;
  step = step ?? 1;

  return { start, stop, step };
}

function handlePotentialNegativeIndex(index: number, dimSize: number) {
  return index < 0 ? dimSize + index : index;
}
