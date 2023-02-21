import { full, randNormal, randUniform } from "../ops/creation";
import { Tensor } from "../tensor";
import { DType } from "../types";

export function makeBlobs<D extends DType.float32 | DType.float64>(
  nSamples: number,
  nFeatures: number,
  centers: number,
  clusterStd: number = 1.0,
  centerBox: number[] = [-10, 10],
  randomState?: number | string,
  dtype: D = DType.float32 as D
): {
  X: Tensor<D>;
  y: Tensor<D>;
} {
  const means = randUniform(
    [centers, nFeatures],
    centerBox[0],
    centerBox[1],
    dtype,
    randomState
  );
  const X = randNormal([nSamples, nFeatures], dtype, randomState).mul_(clusterStd);
  const y = full([nSamples], 1, dtype);

  for (let i = 0; i < centers; i++) {
    let start = i * Math.floor(nSamples / centers);
    let end = (i + 1) * Math.floor(nSamples / centers);
    if (i === centers - 1) end = nSamples;

    X.slice([{ start: start, stop: end }]).add_(means.slice([i]));
    y.slice([{ start: start, stop: end }]).mul_(i);
  }

  return { X, y };
}
