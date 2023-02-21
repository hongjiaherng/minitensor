import * as minitensor from ".";

const { X, y } = minitensor.datasets.makeBlobs(10, 2, 3, 1.0, [-10, 10], 42);

// Fix this bug: Reshape should compute the strides differently
// compute new strides with the new shape, old strides, old offset, and old shape
console.log(X.array(), X.strides); // strides = [2, 1], offset = 0
console.log(X.slice([null, 0]).array(), X.slice([null, 0]).strides); // strides = [2], offset = 0
console.log(
  X.slice([null, 0]).reshape([2, 5]).array(),
  X.slice([null, 0]).reshape([2, 5]).strides
); // strides = [5 * 2, 2], offset = 0

console.log(X.array(), X.strides);  // strides = [2, 1], offset = 1
console.log(X.slice([null, 1]).array(), X.slice([null, 1]).strides); // strides = [2], offset = 1
console.log(
  X.slice([null, 1]).reshape([2, 5]).array(),
  X.slice([null, 1]).reshape([2, 5]).strides
);  // strides = [5 * 2, 2], offset = 1

