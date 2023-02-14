import * as minitensor from ".";

const a = minitensor.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3], "int32");
console.log(a);
console.log(a.array());
console.log();

const sliced = minitensor.slice(a, [{ 0: 2 }, { 1: 3 }]);
console.log(sliced);
console.log(sliced.array());
console.log();

console.log(a);
console.log(a.array());

minitensor.slice(a, [3, null]).set(1000);

console.log(a);
console.log(a.array());

minitensor.slice(a, [null, 2]).set(true);
console.log(a);
console.log(a.array());

minitensor.slice(a, [0, 2]).set(3.542);
console.log(a);
console.log(a.array());



