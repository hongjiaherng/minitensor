import { full, randNormal, randUniform } from "../ops/creation";
import { add } from "../ops/ewise_binary";
import { slice } from "../ops/view";
import { DType } from "../types";

export function makeBlobs(
  nSamples: number,
  nFeatures: number,
  centers: number,
  clusterStd: number = 1.0,
  centerBox: number[] = [-10, 10],
  randomState?: number | string,
  dtype: DType.float32 | DType.float64 = DType.float32
) {
  const means = randUniform(
    [centers, nFeatures],
    centerBox[0],
    centerBox[1],
    dtype,
    randomState
  );
  const X = randNormal([nSamples, nFeatures], dtype, randomState);
  const y = full([nSamples], 1, dtype);

  for (let i = 0; i < centers; i++) {
    let start = i * Math.floor(nSamples / centers);
    let end = (i + 1) * Math.floor(nSamples / centers);
    if (i === centers - 1) end = nSamples;

    slice(X, [{ start: start, end: end }]) += slice(means, [i])
    slice(y, [{ start: start, end: end }]) *= i;
  }
}

// TODO: Fix slice (done)
// TODO: Implement inplace operation +=, *=, -=, /=
// TODO: Implement multiply
// TODO: add related methods to tensor class