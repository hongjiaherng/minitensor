import { Tensor } from "../tensor";
import { empty } from "./tensor_init";
import seedrandom from "seedrandom";

export function randNormal(
	shape: number[],
	mean: number = 0,
	stddev: number = 1,
	dtype: keyof RandDType = "float32",
	randomState?: number | string
): Tensor<keyof RandDType> {
	const tensor = empty(shape, dtype);
	const random = new NormalRandom(mean, stddev, dtype, randomState);
  
	for (let i = 0; i < tensor.data.size; i++) {
		tensor.data.setByIndex(random.next(), i);
	}

	return tensor;
}

export function randUniform(
	shape: number[],
	low: number = 0,
	high: number = 1,
	dtype: keyof RandDType = "float32",
	randomState?: number | string
): Tensor<keyof RandDType> {
	const tensor = empty(shape, dtype);
	const random = new UniformRandom(low, high, dtype, randomState);

	for (let i = 0; i < tensor.data.size; i++) {
		tensor.data.setByIndex(random.next(), i);
	}

	return tensor;
}

export interface RandDType {
	float32: Float32Array;
	float64: Float64Array;
	int32: Int32Array;
	int16: Int16Array;
}

export interface RandomBase {
	next(): number;
}

export class UniformRandom implements RandomBase {
	private low: number;
	private high: number;
	private random: () => number;
	private dtype?: keyof RandDType;

	constructor(
		low = 0,
		high = 1,
		dtype?: keyof RandDType,
		randomState?: number | string
	) {
		this.low = low;
		this.high = high;
		this.dtype = dtype || "float32";
		if (randomState == null) {
			randomState = Math.random();
		}
		if (typeof randomState === "number") {
			randomState = randomState.toString();
		}
		this.random = seedrandom(randomState);

		if (this.low > this.high) {
			throw new Error("low must be <= high");
		}
		if (
			(this.dtype === "int32" || this.dtype === "int16") &&
			this.high - this.low <= 1
		) {
			throw new Error(
				`The difference between ${this.low} - ${this.high} must be > 1 for dtype ${this.dtype}`
			);
		}
	}

	private convertToDType(value: number): number {
		if (this.dtype === "float32" || this.dtype === "float64") {
			return value;
		}
		return Math.round(value);
	}

	next(): number {
		return this.convertToDType(
			this.random() * (this.high - this.low) + this.low
		);
	}
}

export class NormalRandom implements RandomBase {
	private mean: number;
	private stddev: number;
	private random: () => number;
	private dtype?: keyof RandDType;

	constructor(
		mean = 0,
		stddev = 1,
		dtype?: keyof RandDType,
		randomState?: number | string
	) {
		this.mean = mean;
		this.stddev = stddev;
		this.dtype = dtype || "float32";
		if (randomState == null) {
			randomState = Math.random();
		}
		if (typeof randomState === "number") {
			randomState = randomState.toString();
		}
		this.random = seedrandom(randomState);
	}

	private convertToDType(value: number): number {
		if (this.dtype === "float32" || this.dtype === "float64") {
			return value;
		}
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
