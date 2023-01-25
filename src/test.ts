import { broadcastShapes, broadcastTensors, broadcastTo, computeBroadcastedStrides } from "./ops/broadcast";
import { tensor } from "./tensor_init";
import { Tensor } from "./tensor";
import { DataType } from "./types";
import { determineDType } from "./utils";

const a = tensor([1.123, 2.123, 3.123, 4.211, 5.12134, 6.124], [2, 3], "float32");
const b = tensor([10, 20, 30], [3], "int32");
const c = broadcastTo(b, [4, 3]);

console.log(a);
console.log(a.view());
console.log(b);
console.log(b.view());
console.log(c)
console.log(c.view())

// const iterate = (tensor: Tensor<DataType>) => {
//   for (let i = 0; i < tensor.shape[0]; i++) {
//     for (let j = 0; j < tensor.shape[1]; j++) {
//       for (let k = 0; k < tensor.shape[2]; k++) {
//         console.log(i, j, k, i * tensor.strides[0] + j * tensor.strides[1] + k * tensor.strides[2])
//       }
//     }
//     console.log()
//   }
// }

// iterate(c);

// let index = 0;
// let offset = 0
// while (index < c.size) {
//   offset = offset + 
//   // 0, 1, 2, 0, 1, 2

//   console.log(c.data[offset]);

//   index++;
// }

// console.log(c);