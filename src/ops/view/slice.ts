import assert from "assert";
import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { asStrided } from "../creation";
import { squeeze_ } from "./squeeze";

interface DimensionSlicingSelection {
  [start: number]: number | string | null;
  step?: number;
}
type DimensionSelection = DimensionSlicingSelection | number | null;
type SlicingSelection = DimensionSelection[];

export function slice<D extends DType>(
  x: Tensor<D>,
  selection: SlicingSelection,
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
   * 2) beginIndices, slicedShape, slicedStrides -> slicedOffset
   *
   * 3) slicedStrides, slicedOffset, slicedShape -> new tensor view
   */

  assertSlicingSelectionIsValid(selection, x.shape);

  // turn selection into beginIndices, slicedShape, and slicedStrides
  const slicedShape = new Array(x.shape.length);
  const beginIndices = new Array(x.shape.length);
  const slicedStrides = new Array(x.shape.length);

  for (let i = 0; i < x.shape.length; i++) {
    if (selection[i] == null) {
      // null or undefined
      beginIndices[i] = 0;
      slicedShape[i] = x.shape[i];
      slicedStrides[i] = x.strides[i];
    } else if (typeof selection[i] === "number") {
      beginIndices[i] = handleStartIndex(selection[i] as number, x.shape[i]);
      slicedShape[i] = 1;
      slicedStrides[i] = x.strides[i];
    } else {
      const [start, end] = getStartAndEndIndex(
        selection[i] as DimensionSlicingSelection,
        x.shape[i]
      ); // handle negative or null start and end index
      const step = (selection[i] as DimensionSlicingSelection).step ?? 1;
      beginIndices[i] = step > 0 ? start : end - 1;
      slicedShape[i] = Math.ceil((end - start) / Math.abs(step));
      slicedStrides[i] = x.strides[i] * step;
    }
  }

  // calculate new offset using beginIndices and old strides
  const slicedOffset =
    x.offset +
    beginIndices.reduce((acc, beginI, i) => acc + beginI * x.strides[i], 0);

  // create new tensor view with new strides, offset, and shape
  return keepDim
    ? asStrided(x, slicedShape, slicedStrides, slicedOffset)
    : squeeze_(asStrided(x, slicedShape, slicedStrides, slicedOffset));
}

function getStartAndEndIndex(
  dimSelection: DimensionSlicingSelection,
  dimSize: number
): [number, number] {
  const [start, end] = Object.entries(dimSelection)[0].map((v, i) => {
    if (i === 0 && v === null) return 0; // start index is 0 if not specified
    if (v === null) return dimSize; // end index is dimSize if not specified
    return i === 0
      ? handleStartIndex(Number(v), dimSize) // start index, i === 0
      : handleEndIndex(Number(v), dimSize); // end index, i === 1
  });
  return [start, end];
}

function handlePotentialNegativeIndex(index: number, dimSize: number): number {
  if (index < 0) return dimSize + index;
  return index;
}

function handleStartIndex(start: number, dimSize: number): number {
  /**
   * start index is inclusive
   * invalid when:
   * 1) start < -dimSize
   * 		- -4 < -3 (true) => invalid
   * 		- -3 < -3 (false) => valid
   * 		- -2 < -3 (false) => valid
   * 2) start > dimSize - 1
   * 		- 3 > 3-1 (true) => invalid
   * 		- 2 > 3-1 (false) => valid
   * 		- 1 > 3-1 (false) => valid
   *
   */
  if (start < -dimSize || start > dimSize - 1)
    throw new RangeError(
      `Index ${start} is out of bounds for dimension with size ${dimSize}`
    );
  return handlePotentialNegativeIndex(start, dimSize);
}

function handleEndIndex(end: number, dimSize: number): number {
  /**
   * end index is exclusive
   * invalid when:
   * 1) end < -dimSize - 1
   * 		- -5 < -3-1 (true) => invalid
   * 		- -4 < -3-1 (false) => valid
   * 		- -3 < -3-1 (false) => valid
   * 2) end > dimSize
   * 		- 4 > 3 (true) => invalid
   * 		- 3 > 3 (false) => valid
   * 		- 2 > 3 (false) => valid
   *
   */
  if (end < -dimSize - 1 || end > dimSize)
    throw new RangeError(
      `Index ${end} is out of bounds for dimension with size ${dimSize}`
    );
  return handlePotentialNegativeIndex(end, dimSize);
}

function assertSlicingSelectionIsValid(
  selection: SlicingSelection,
  shape: number[]
): void {
  assert(
    selection.length <= shape.length,
    `Too many indices for tensor: tensor is ${shape.length}-dimensional, but ${selection.length} dimensions were indexed`
  );
  assert(
    selection.length > 0,
    "No index was provided for tensor, at least 1 index is required"
  );
  selection.forEach((dimSelection, i) => {
    if (dimSelection !== null && typeof dimSelection === "object") {
      const [start, end] = getStartAndEndIndex(
        dimSelection as DimensionSlicingSelection,
        shape[i]
      );
      assert(
        start < end,
        `Invalid slice selection: start index ${start} must be smaller than end index ${end}`
      );

      assert(
        (dimSelection as DimensionSlicingSelection).step !== 0,
        "Step must be non-zero"
      );
    }
  });
}
