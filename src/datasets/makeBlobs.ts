import { randNormal, randUniform } from "../ops/random";
import { empty, full } from "../ops/tensor_init";
import { Storage } from "../storage";
import { Tensor } from "../tensor";

export function makeBlobs(
	nSamples: number,
	nFeatures: number,
	centers: number,
	clusterStd: number = 1.0,
	centerBox: number[] = [-10, 10],
	randomState?: number | string,
	dtype: "float32" | "float64" = "float32"
): { X: Tensor<"float32" | "float64">; y: Tensor<"float32" | "float64"> } {
	const X = empty([nSamples, nFeatures], dtype);
	const y = empty([nSamples], dtype);

	for (let i = 0; i < centers; i++) {
		let nSamplesPerCenter = Math.floor(nSamples / centers);
		if (i === centers - 1) {
			nSamplesPerCenter += nSamples % centers;
		}

		// generate mean of cluster uniformly within centerBox for each feature
		const mean = randUniform(
			[nFeatures],
			centerBox[0],
			centerBox[1],
			dtype,
			randomState
		);

		// generate nSamplesPerCenter samples for cluster i from a normal distribution with mean and clusterStd
		for (let j = 0; j < nFeatures; j++) {
			const x_ = randNormal(
				[nSamplesPerCenter],
				mean.data.get([j]),
				clusterStd,
				dtype,
				randomState
			);
			X.data.setBySlice(
				x_.data as Storage<"float32" | "float64">,
				[i * Math.floor(nSamples / centers), j],
				[nSamplesPerCenter, 1]
			);
			y.data.setBySlice(
				full([nSamplesPerCenter], i).data as Storage<"float32" | "float64">,
				[i * Math.floor(nSamples / centers)],
				[nSamplesPerCenter]
			);
		}
	}

	return { X, y };
}
