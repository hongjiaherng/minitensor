import * as minitensor from ".";

const tensor = minitensor.reshape(minitensor.arange(1, 13), [4, 3]);

let sliced;

// multiple tries of slice with invalid selection
// sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 0 }, 0]);
// console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 1 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 2 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 3 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 4 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 5 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 6 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 7 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 8 }, 0]);
console.log(sliced.array())
sliced = minitensor.slice(tensor, [{ start: 0, stop: 4, step: 9 }, 0]);
console.log(sliced.array())




