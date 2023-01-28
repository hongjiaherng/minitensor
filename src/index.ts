import { makeBlobs } from "./datasets/makeBlobs";
import { _broadcastTo } from "./ops/broadcast";
import { UniformRandom } from "./ops/random";
import { arange, tensor } from "./ops/tensor_init";
import { Storage } from "./storage";
import { castToTypedArray } from "./utils";



const { X, y } = makeBlobs(10, 2, 3)

console.log(X.view())
console.log(y.view())

// const a = tensor(
// 	[
// 		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
// 		22, 23, 24,
// 	],
// 	[4, 3, 2]
// );

// const a = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
// const b = tensor([10, 10, 10, 10, 10, 10, 10, 10, 10], [3, 3])
// const c = a.add(b);
// console.log(c);

// const a = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
// const b = tensor([10, 20, 30], [3]);
// a.mul(b);
// console.log(a.add([10, 20]))

// const a = tensor(1); // target float32
// const b = tensor(true); // target bool
// const c = tensor([1, 2, 3, 4]); // target float32
// const d = tensor([true, false, true, false]); // target bool
// const g = tensor(new Uint8Array([1, 0, 1, 0]));  // target bool
// const i = tensor(new Int16Array([1, 2, 3, 4]));  // target int16
// const e = tensor(new Int32Array([1, 2, 3, 4]));  // target int32
// const f = tensor(new Float32Array([1, 2, 3, 4]));  // target float32
// const h = tensor(new Float64Array([1, 2, 3, 4]));  // target float64

// const a = castToTypedArray(1, "float32");
// const b = castToTypedArray(true, "bool");
// const c = castToTypedArray([1, 2, 3, 4], "float32");
// const d = castToTypedArray([true, false, true, false], "bool");
// const g = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "bool");
// const i = castToTypedArray(new Int16Array([1, 2, 3, 4]), "int16");
// const e = castToTypedArray(new Int32Array([1, 2, 3, 4]), "int32");
// const f = castToTypedArray(new Float32Array([1, 2, 3, 4]), "float32");
// const h = castToTypedArray(new Float64Array([1, 2, 3, 4]), "float64");

// const a = castToTypedArray(1, "float32");
// const b = castToTypedArray(1, "float64");
// const c = castToTypedArray(1, "int16");
// const d = castToTypedArray(1, "int32");
// const e = castToTypedArray(1, "bool");

// const a = castToTypedArray(true, "float32");
// const b = castToTypedArray(true, "float64");
// const c = castToTypedArray(true, "int16");
// const d = castToTypedArray(true, "int32");
// const e = castToTypedArray(true, "bool");

// const a = castToTypedArray([true, false], "float32");
// const b = castToTypedArray([true, false], "float64");
// const c = castToTypedArray([true, false], "int16");
// const d = castToTypedArray([true, false], "int32");
// const e = castToTypedArray([true, false], "bool");

// const a = castToTypedArray([1, 2, 3, 4], "float32");
// const b = castToTypedArray([1, 2, 3, 4], "float64");
// const c = castToTypedArray([1, 2, 3, 4], "int16");
// const d = castToTypedArray([1, 2, 3, 4], "int32");
// const e = castToTypedArray([1, 2, 3, 4], "bool");

// const a = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "float32");
// const b = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "float64");
// const c = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "int16");
// const d = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "int32");
// const e = castToTypedArray(new Uint8Array([1, 0, 1, 0]), "bool");

// const a = castToTypedArray(new Int16Array([1, 2, 3, 4]), "float32");
// const b = castToTypedArray(new Int16Array([1, 2, 3, 4]), "float64");
// const c = castToTypedArray(new Int16Array([1, 2, 3, 4]), "int16");
// const d = castToTypedArray(new Int16Array([1, 2, 3, 4]), "int32");
// const e = castToTypedArray(new Int16Array([1, 2, 3, 4]), "bool");

// const a = castToTypedArray(new Int32Array([1, 2, 3, 4]), "float32");
// const b = castToTypedArray(new Int32Array([1, 2, 3, 4]), "float64");
// const c = castToTypedArray(new Int32Array([1, 2, 3, 4]), "int16");
// const d = castToTypedArray(new Int32Array([1, 2, 3, 4]), "int32");
// const e = castToTypedArray(new Int32Array([1, 2, 3, 4]), "bool");

// const a = castToTypedArray(new Float32Array([1, 2, 3, 4]), "float32");
// const b = castToTypedArray(new Float32Array([1, 2, 3, 4]), "float64");
// const c = castToTypedArray(new Float32Array([1, 2, 3, 4]), "int16");
// const d = castToTypedArray(new Float32Array([1, 2, 3, 4]), "int32");
// const e = castToTypedArray(new Float32Array([1, 2, 3, 4]), "bool");

// const a = castToTypedArray(new Float64Array([1, 2, 3, 4]), "float32");
// const b = castToTypedArray(new Float64Array([1, 2, 3, 4]), "float64");
// const c = castToTypedArray(new Float64Array([1, 2, 3, 4]), "int16");
// const d = castToTypedArray(new Float64Array([1, 2, 3, 4]), "int32");
// const e = castToTypedArray(new Float64Array([1, 2, 3, 4]), "bool");
