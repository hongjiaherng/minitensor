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


console.log(minitensor.computeStrides([32, 32]))
console.log(minitensor.computeStrides([4, 32, 8]))

const computeReshapedStrides = () => {
  // Example 1:
  // we have: 
  // oldStrides = [2]
  // oldShape = [10]
  // newShape = [2, 5]

  // we want:
  // newStrides = [5 * 2, 2]

  // Example 2:
  // we have:
  // oldStrides = [2, 1]
  // oldShape = [10, 2]
  // newShape = [4, 5]

  // we want:
  // newStrides = [5 * 4, 4]


  // Example 3:
  // we have:
  // oldStrides = [2, 1]
  // oldShape = [10, 2]
  // newShape = [4, 5, 1]

  // we want:
  // newStrides = 
}