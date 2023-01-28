import {
	broadcastShapes,
	broadcastTensors,
	broadcastTo,
	computeBroadcastedStrides,
	TensorsIterator,
} from "./ops/broadcast";
import { arange, tensor } from "./tensor_init";
import { Tensor } from "./tensor";
import { DataType } from "./types";
import { determineDType } from "./utils";
import { add } from "./ops";

let a, b, c, scalar;

a = arange(0, 9).reshape([3, 3]);
b = arange(0, 3).reshape([3]);
c = a.add(b);
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

a = arange(0, 3);
scalar = 2.0;
c = a.add(scalar);
console.log(`[${a.shape}] + [1] = [${c.shape}]`);

a = arange(0, 5 * 4).reshape([5, 4]);
b = tensor([1])
c = a.add(b)
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

a = arange(0, 5 * 4).reshape([5, 4]);
b = arange(0, 4);
c = a.add(b);
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

a = arange(0, 15 * 3 * 5).reshape([15, 3, 5]);
b = arange(0, 15 * 1 * 5).reshape([15, 1, 5]);
c = a.add(b)
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

a = arange(0, 15 * 3 * 5).reshape([15, 3, 5]);
b = arange(0, 3 * 5).reshape([3, 5]);
c = a.add(b)
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

a = arange(0, 15 * 3 * 5).reshape([15, 3, 5]);
b = arange(0, 3).reshape([3, 1])
c = a.add(b)
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);

try {
  a = arange(0, 3)
  b = arange(0, 4)
  c = a.add(b)
  console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);
} catch (error) {
  console.log((error as Error).message)
}

try {
  a = arange(0, 2 * 1).reshape([2, 1])
  b = arange(0, 8 * 4 * 3).reshape([8, 4, 3])
  c = a.add(b)
  console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);
} catch (error) {
  console.log((error as Error).message)
}

a = arange(0, 10 * 3).reshape([10, 3])
b = arange(0, 5 * 1 * 3).reshape([5, 1, 3])
c = a.add(b)
console.log(`[${a.shape}] + [${b.shape}] = [${c.shape}]`);
