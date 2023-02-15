import { DType, NumericDType } from "../../types";
import seedrandom from "seedrandom";
import { empty } from "./empty";

export function randNormal(
  shape: number[],
  mean: number = 0,
  std: number = 1,
  dtype: NumericDType = DType.float32,
  randomState?: number | string
) {
  const rand = new NormalRandom(mean, std, dtype, randomState);
  const tensor = empty(shape, dtype);
  for (let i = 0; i < tensor.size; i++) {
    tensor._setByIndex(i, rand.next());
  }
  return tensor;
}

export function randUniform(
  shape: number[],
  low: number = 0,
  high: number = 1,
  dtype: NumericDType = DType.float32,
  randomState?: number | string
) {
  const rand = new UniformRandom(low, high, dtype, randomState);
  const tensor = empty(shape, dtype);
  for (let i = 0; i < tensor.size; i++) {
    tensor._setByIndex(i, rand.next());
  }
  return tensor;
}

class UniformRandom {
  private low: number;
  private high: number;
  private random: () => number;
  private dtype?: NumericDType;

  constructor(
    low = 0,
    high = 1,
    dtype?: NumericDType,
    randomState?: number | string
  ) {
    this.low = low;
    this.high = high;
    this.dtype = dtype ?? DType.float32;
    if (randomState == null) randomState = Math.random();
    if (typeof randomState === "number") randomState = randomState.toString();
    this.random = seedrandom(randomState);

    if (this.low > this.high) {
      throw new Error("low must be <= high");
    }
    if (
      (this.dtype === DType.int32 || this.dtype === DType.int16) &&
      this.high - this.low <= 1
    ) {
      throw new Error(
        `The difference between ${this.low} - ${this.high} must be > 1 for dtype ${this.dtype}`
      );
    }
  }

  private convertToDType(value: number): number {
    if (this.dtype === DType.float32 || this.dtype === DType.float64)
      return value;
    return Math.round(value);
  }

  next(): number {
    return this.convertToDType(
      this.random() * (this.high - this.low) + this.low
    );
  }
}

class NormalRandom {
  private mean: number;
  private stddev: number;
  private random: () => number;
  private dtype?: NumericDType;

  constructor(
    mean = 0,
    stddev = 1,
    dtype?: NumericDType,
    randomState?: number | string
  ) {
    this.mean = mean;
    this.stddev = stddev;
    this.dtype = dtype ?? DType.float32;

    if (randomState == null) randomState = Math.random();
    if (typeof randomState === "number") randomState = randomState.toString();

    this.random = seedrandom(randomState);
  }

  private convertToDType(value: number): number {
    if (this.dtype === DType.float32 || this.dtype === DType.float64)
      return value;
    return Math.round(value);
  }

  next(): number {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    let u = 0,
      v = 0;
    while (u === 0) u = this.random();
    while (v === 0) v = this.random();

    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return this.convertToDType(z * this.stddev + this.mean);
  }
}
