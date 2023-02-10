import * as minitensor from ".";

// const a = minitensor.tensor([1, 2, 3]);
// const b = minitensor.broadcast.broadcastTo(a, [3, 3]);

const a = minitensor.reshape(minitensor.arange(0, 9), [3, 3])
console.log(a)
console.log(a.array())

const b = minitensor.slice(a, [0, 0], [1, 3])
console.log(b);
console.log(b.array())

const c = minitensor.slice(a, [0, 0], [3, 2])
console.log(c);
console.log(c.array())

const d = minitensor.slice(a, [1, 1], [2, 2])
console.log(d);
console.log(d.array())