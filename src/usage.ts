import * as minitensor from ".";

const a = minitensor.fromTensorLike(
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  [4, 3]
);

// TODO: unit test for slice
// TODO: cover -ve step slicing
// TODO: option to flatten the sliced tensor into 1D

// // Single element indexing
// // Slicing

// // 2D tensor
// // slice a row
// // a[0], // a[0, :]
// minitensor.slice(a, [0]);
// minitensor.slice(a, [-4]);
// // a[0, 0:3]
// minitensor.slice(a, [0, {0:3}])
// minitensor.slice(a, [-4, {"-3":3}])
// minitensor.slice(a, [0, {0:null}])
// minitensor.slice(a, [0, null])
// minitensor.slice(a, [-4, {"-3":null}])
// minitensor.slice(a, [0, null])
// // a[0:1, 0:3]
// minitensor.slice(a, [{0:1}, {0:3}])
// minitensor.slice(a, [{0:1}, {0:null}])
// minitensor.slice(a, [{0:1}, null])
// minitensor.slice(a, [{"-4": -3}, {"-3":3}])
// minitensor.slice(a, [{0:1}, null])

// console.log("\n")

// // selection -> begin and shape
// // begin = [0, 0]
// // shape = [1, 3]

// // expected tensor's attributes:
// // shape = [1, 3]
// // strides = [3, 1]
// // offset = 0

// // slice a column
// // a[:, 0] === a[0:3, 0]
// minitensor.slice(a, [{0:4}, 0])
// minitensor.slice(a, [{0:null}, 0])
// minitensor.slice(a, [{"-4":4}, -3])
// // a[:, 0:1]
// minitensor.slice(a, [{0:4}, {0:1}])
// minitensor.slice(a, [{0:null}, {0:1}])
// minitensor.slice(a, [{"-4":4}, {"-3":-2}])

// console.log("\n")
// // expected tensor's attributes:
// // shape = [4, 1]
// // strides = [3, 1]
// // offset = 0

// // select an entry
// // a[0, 0]
// minitensor.slice(a, [0, 0])
// minitensor.slice(a, [0, {0:1}])
// minitensor.slice(a, [{0:1}, 0])
// minitensor.slice(a, [{0:1}, {0:1}])
// minitensor.slice(a, [-4, -3])

// console.log("\n")
// // expected tensor's attributes:
// // shape = [1, 1]
// // strides = [3, 1]
// // offset = 0

// // slice a shape
// // a[0:2, 0:2]
// minitensor.slice(a, [{1:3}, {0:2}])
// minitensor.slice(a, [{"-3":-1}, {"-3":-1}])

// console.log("\n")
// // expected tensor's attributes:
// // shape = [2, 2]
// // strides = [3, 1]
// // offset = 0

// slice with step
// a[0:4:2, 0:3:1]
// minitensor.slice(a, [{0:4, step:2}, {0:3, step:1}])
// selection -> begin and shape
// begin = [0, 0]
// shape = [2, 3]

// expected tensor's attributes:
// shape = [2, 3]
// strides = [6, 1]
// offset = 0
// tensor = [[1, 2, 3], [7, 8, 9]]

// a[0:4:2, 0:3:2]
// minitensor.slice(a, [{0:4, step:2}, {0:3, step:2}])
// selection -> begin and shape
// begin = [0, 0]
// shape = [2, 2]

// expected tensor's attributes:
// shape = [2, 2]
// strides = [6, 2]
// offset = 0
// tensor = [[1, 3], [7, 9]]

// a[0:4:2, 0:3:3]
// minitensor.slice(a, [{0:4, step:2}, {0:3, step:3}])
// selection -> begin and shape
// begin = [0, 0]
// shape = [2, 1]

// expected tensor's attributes:
// shape = [2, 1]
// strides = [6, 3]
// offset = 0
// tensor = [[1], [7]]

// a[1:4:2, 0:3:3]
// minitensor.slice(a, [{1:4, step:2}, {0:3, step:3}])
// selection -> begin and shape
// begin = [1, 0]
// shape = [2, 1]

// expected tensor's attributes:
// shape = [2, 1]
// strides = [6, 3]
// offset = 3
// tensor = [[4], [10]]

// a[0:4:-1, :]
minitensor.slice(a, [{ 0: 4, step: -1 }, null]);
// selection -> begin and shape
// begin = [3, 0]
// shape = [4, 3]
// reverse the order of the first dimension

// expected tensor's attributes:
// shape = [4, 3]
// strides = [-3, 1]
// offset = 9
// tensor = [[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]]

// a[0:4:-2, :]
minitensor.slice(a, [{ 0: 4, step: -2 }, null]);
// selection -> begin and shape
// begin = [3, 0]
// shape = [2, 3]

// expected tensor's attributes:
// shape = [2, 3]
// strides = [-6, 1]
// offset = 9
// tensor = [[10, 11, 12], [4, 5, 6]]
