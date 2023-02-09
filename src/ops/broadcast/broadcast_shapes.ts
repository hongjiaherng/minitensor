export function broadcastShapes(
	inputShape: number[],
	otherShape: number[]
): number[] {
	const inputShape_ = [...inputShape]; // make deepcopy
	const otherShape_ = [...otherShape];
	const resultShape = [];

	// pad with 1s if shapes are not the same length
	while (inputShape_.length != otherShape_.length) {
		if (inputShape_.length < otherShape_.length) {
			inputShape_.unshift(1);
		} else {
			otherShape_.unshift(1);
		}
	}

	for (let i = inputShape_.length - 1; i >= 0; i--) {
		if (
			inputShape_[i] !== otherShape_[i] &&
			inputShape_[i] !== 1 &&
			otherShape_[i] !== 1
		) {
			throw new Error(
				`Incompatible shapes to be braodcasted together: [${inputShape}] and [${otherShape}]`
			);
		}
		resultShape.unshift(Math.max(inputShape_[i], otherShape_[i]));
	}

	return resultShape;
}
