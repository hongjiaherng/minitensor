import { Tensor } from "../tensor";
import { tensor, empty } from "../tensor_init";
import { DataType, TensorLike } from "../types";
import { determineDType, isEqualShape } from "../utils";
import { broadcastTensors } from "./broadcast";

export function add(
	a: Tensor<DataType> | TensorLike,
	b: Tensor<DataType> | TensorLike
): Tensor<DataType> {
	const tensorA =
		typeof a === "number" || typeof a === "boolean"
			? tensor(a, [1], "float32")
			: (a as Tensor<DataType>);
	const tensorB =
		typeof b === "number" || typeof b === "boolean"
			? tensor(b, [1], "float32")
			: (b as Tensor<DataType>);

	let resultTensor;
	if (isEqualShape(tensorA.shape, tensorB.shape)) {
		// do the element-wise addition
		resultTensor = empty(
			Array.from(tensorA.shape),  // make a deepcopy of the shape array
			determineDType(tensorA.dtype, tensorB.dtype)  // determine the dtype of the result
		);
		for (let i = 0; i < resultTensor.size; i++) {
			resultTensor.data[i] = tensorA.data[i] + tensorB.data[i];
		}
	} else {
    console.log("broadcasting tensors");
    const [bcTensorA, bcTensorB] = broadcastTensors(tensorA, tensorB);
    resultTensor = empty(
      Array.from(bcTensorA.shape),  // make a deepcopy of the shape array
      determineDType(bcTensorA.dtype, bcTensorB.dtype)  // determine the dtype of the result
    )
    console.log(bcTensorA);
    console.log(bcTensorB);
    
    for (let i = 0; i < resultTensor.size; i++) {
      resultTensor.data[i] = bcTensorA.data[i] + bcTensorB.data[i];
    }

	}
	return resultTensor;
}
