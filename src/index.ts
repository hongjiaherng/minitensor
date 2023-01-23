import { tensor } from "./tensor_init";

// init with:
// Uint8Array -> Float32Array
// Int32Array -> no change
// Float32Array -> no change
// number[] -> Float32Array
// boolean[] -> Uint8Array

let t;
let numArr = [1.123, 2.123, 3.631];
let boolArr = [true, false, true];
let float32Arr = Float32Array.from([1.123, 2.123, 3.631]);
let int32Arr = Int32Array.from([1, 2, 3]);
let uint8Arr = Uint8Array.from([1, 2, 3]);

// Test tensor initialization with different data types (dtype inferred)
console.log(
	"=== (START) Test tensor initialization Test tensor initialization with different data types (dtype inferred) ===\n"
);
console.log("Init with number[]; Expected result: Float32Array");
t = tensor(numArr);
console.log("Original array:", numArr);
console.log(t, "\n");

console.log(
	"Init with TypedArray (Float32Array); Expected result: Float32Array"
);
t = tensor(float32Arr);
console.log("Original array:", float32Arr);
console.log(t, "\n");

console.log("Init with TypedArray (Uint8Array); Expected result: Int32Array");
t = tensor(uint8Arr);
console.log("Original array:", uint8Arr);
console.log(t, "\n");

console.log("Init with TypedArray (Int32Array); Expected result: Int32Array");
t = tensor(int32Arr);
console.log("Original array:", int32Arr);
console.log(t, "\n");

console.log("Init with bool[]; Expected result: Uint8Array");
t = tensor(boolArr);
console.log("Original array:", boolArr);
console.log(t, "\n");

console.log(
	"=== (END) Test tensor initialization Test tensor initialization with different data types (dtype inferred) ===\n"
);

// Test tensor initialization with different data types (dtype specified)
console.log(
	"=== (START) Test tensor initialization with different data types (dtype specified) ===\n"
);
// float32
console.log(
	"Init with number[], specify float32; Expected result: Float32Array"
);
t = tensor(numArr, [3], "float32");
console.log("Original array:", numArr);
console.log(t, "\n");

try {
	console.log(
		"Init with bool[], specify float32; Expected result: Error thrown"
	);
	t = tensor(boolArr, [3], "float32");
} catch (error) {
	console.log("Original array:", boolArr);
	console.log((error as Error).message, "\n");
}

console.log(
	"Init with TypedArray (Float32Array), specify float32; Expected result: Float32Array"
);
t = tensor(float32Arr, [3], "float32");
console.log("Original array:", float32Arr);
console.log(t, "\n");

console.log(
	"Init with TypedArray (Int32Array), specify float32; Expected result: Float32Array"
);
t = tensor(int32Arr, [3], "float32");
console.log("Original array:", int32Arr);
console.log(t, "\n");

console.log(
	"Init with TypedArray (Uint8Array), specify float32; Expected result: Float32Array"
);
t = tensor(uint8Arr, [3], "float32");
console.log("Original array:", uint8Arr);
console.log(t, "\n");

// int32
console.log("Init with number[], specify int32; Expected result: Int32Array");
t = tensor(numArr, [3], "int32");
console.log("Original array:", numArr);
console.log(t, "\n");

try {
	console.log("Init with bool[], specify int32; Expected result: Error thrown");
	t = tensor(boolArr, [3], "int32");
} catch (error) {
	console.log("Original array:", boolArr);
	console.log((error as Error).message, "\n");
}

console.log(
	"Init with TypedArray (Float32Array), specify int32; Expected result: Int32Array"
);
t = tensor(float32Arr, [3], "int32");
console.log("Original array:", float32Arr);
console.log(t, "\n");

console.log(
	"Init with TypedArray (Int32Array), specify int32; Expected result: Int32Array"
);
t = tensor(int32Arr, [3], "int32");
console.log("Original array:", int32Arr);
console.log(t, "\n");

console.log(
	"Init with TypedArray (Uint8Array), specify int32; Expected result: Int32Array"
);
t = tensor(uint8Arr, [3], "int32");
console.log("Original array:", uint8Arr);
console.log(t, "\n");

// bool
try {
	console.log(
		"Init with number[], specify bool; Expected result: Error thrown"
	);
	t = tensor(numArr, [3], "bool");
} catch (error) {
	console.log("Original array:", numArr);
	console.log((error as Error).message, "\n");
}

console.log("Init with bool[], specify bool; Expected result: Uint8Array");
t = tensor(boolArr, [3], "bool");
console.log("Original array:", boolArr);
console.log(t, "\n");

try {
	console.log(
		"Init with TypedArray (Float32Array), specify bool; Expected result: Error thrown"
	);
	t = tensor(float32Arr, [3], "bool");
} catch (error) {
	console.log("Original array:", float32Arr);
	console.log((error as Error).message, "\n");
}

try {
	console.log(
		"Init with TypedArray (Int32Array), specify bool; Expected result: Error thrown"
	);
	t = tensor(int32Arr, [3], "bool");
} catch (error) {
	console.log("Original array:", int32Arr);
	console.log((error as Error).message, "\n");
}

// Uint8Array doesn't contain just 0 and 1, so it's not a valid bool array
try {
	console.log(
		"Init with TypedArray (Uint8Array) with mixed numbers, specify bool; Expected result: Error thrown"
	);
	t = tensor(uint8Arr, [3], "bool");
} catch (error) {
	console.log("Original array:", uint8Arr);
	console.log((error as Error).message, "\n");
}

console.log(
	"Init with TypedArray (Uint8Array) with only 0 and 1, specify bool; Expected result: Uint8Array"
);
t = tensor(Uint8Array.from([0, 0, 1]), [3], "bool");
console.log("Original array:", Uint8Array.from([0, 0, 1]));
console.log(t, "\n");

console.log("=== (END) Test tensor initialization with different data types (dtype specified) ===\n");

t = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 7]);
console.log(t);