import * as minitensor from ".";
import { upcastType } from "./types_util";

const a = minitensor.tensor(
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  [4, 3],
  "float32"
);
a.clone();
a.type("float32");
a.type("int32");
a.dtype;
a.data;

const b = minitensor.tensor([0, 1, 0], undefined, "bool");
b.clone();
b.type("float32");
b.type("int32");
b.type("bool");

const c = minitensor.add(a, b)
console.log(c);
console.log(c.array())

const d = minitensor.add([1, 2, 3], [4, 5, 6])
console.log(d)
console.log(d.array())


// const int32Tensor = minitensor.tensor(
//   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
//   [4, 3],
//   "int32"
// );
// console.log(int32Tensor.array());
// int32Tensor._setByIndex(6, 1000.12345);
// console.log(int32Tensor.array());

// int32Tensor.set([true, false, true])
// console.log(int32Tensor.array());

// const boolTensor = minitensor.tensor(
//   [
//     true,
//     false,
//     true,
//     false,
//     true,
//     false,
//     true, // 6th
//     false,
//     true,
//     false,
//     true,
//     false
//   ],
//   [4, 3],
//   "bool"
// );

// console.log(boolTensor.array());
// boolTensor._setByIndex(6, false);
// console.log(boolTensor.array());
// boolTensor.set([100, 200, 300])

// console.log(boolTensor.array());

// console.log(a);
// console.log(a.array());
// console.log();

// const sliced = minitensor.slice(a, [{ 0: 2 }, { 1: 3 }]);
// console.log(sliced);
// console.log(sliced.array());
// console.log();

// console.log(a);
// console.log(a.array());

// minitensor.slice(a, [3, null]).set(1000);

// console.log(a);
// console.log(a.array());

// minitensor.slice(a, [null, 2]).set(true);
// console.log(a);
// console.log(a.array());

// minitensor.slice(a, [0, 2]).set(3.542);
// console.log(a);
// console.log(a.array());
