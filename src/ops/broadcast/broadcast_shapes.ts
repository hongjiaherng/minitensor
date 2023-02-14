import { assertValidShape } from "../../shape_strides_util";

export function broadcastShapes(...shapes: number[][]): number[] {
  shapes.forEach((shape) => assertValidShape(shape)); // validate shapes
  if (shapes.length < 2) return shapes[0]; // no need to broadcast if there is only one shape, return the shape directly, undefined if shapes is empty

  // find the max rank
  const maxRank = Math.max(...shapes.map((shape) => shape.length));
  const newShape = new Array<number>(maxRank);

  for (let i = 0; i < maxRank; i++) {
    const dimSizes = shapes.reduce((dimSizes, shape) => {
      const dimSize = shape.at(-(i + 1)); // index from the end, e.g. -1 is the last element; return undefined if out of bound
      if (dimSize !== undefined) dimSizes.add(dimSize);
      return dimSizes;
    }, new Set([1]));

    // if there are more than 2 unique dim sizes, i.e., {1, dim_1, dim_2}, then the shapes are not broadcastable
    if (dimSizes.size > 2) {
      throw new Error(
        `Incompatible shapes to be broadcasted together: ${shapes.map(
          (shape) => "[" + shape + "]"
        )}`
      );
    }
    newShape[maxRank - 1 - i] = Math.max(...dimSizes);
  }
  return newShape;
}
