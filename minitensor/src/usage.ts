import * as mt from ".";

const a = mt.arange(0, 24).reshape([2, 3, 4]);
console.log(a);

console.log(a.array());
console.log();

const aView = a.slice([0, 1, { start: 0, stop: 4 }]);

console.log(aView);
console.log(aView.array());
console.log();

console.log(mt.isContiguous(a));
console.log(mt.isContiguous(aView));

const aViewReshapedView = aView.reshape([2, 2]);
console.log(aViewReshapedView);
console.log(aViewReshapedView.array());
console.log(mt.isContiguous(aViewReshapedView));
