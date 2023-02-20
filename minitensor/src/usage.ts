import * as minitensor from ".";

const { X, y } = minitensor.datasets.makeBlobs(10, 2, 3, 1.0, [-10, 10], 42)

console.log(X)
console.log(y)

