import { describe, expect, test } from "@jest/globals";
import { Storage } from "./storage";

describe("Storage initialization with data but no dtype specified", () => {
	test('input: typeof data === "number"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(1, undefined);
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage([1, 2, 3], undefined);
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: typeof data === "boolean"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
		const storage = new Storage(true, undefined);
		expect(storage.dtype).toBe("bool");
		expect(storage.storage).toBeInstanceOf(Uint8Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
		const storage = new Storage([true, false, true], undefined);
		expect(storage.dtype).toBe("bool");
		expect(storage.storage).toBeInstanceOf(Uint8Array);
	});

	test('input: data instanceof Uint8Array && data.every((value) => value === 0 || value === 1); expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 1]), undefined);
		expect(storage.dtype).toBe("bool");
		expect(storage.storage).toBeInstanceOf(Uint8Array);
	});

	test("input: data instanceof Uint8Array && data.some((value) => value !== 0 && value !== 1); expect: throw Error", () => {
		expect(
			() => new Storage(new Uint8Array([0, 1, 0, 2]), undefined)
		).toThrowError();
	});

	test('input: data instanceof Float32Array; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Float32Array([1, 2, 3]), undefined);
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Float64Array; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Float64Array([1, 2, 3]), undefined);
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Int16Array; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Int16Array([1, 2, 3]), undefined);
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Int32Array; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Int32Array([1, 2, 3]), undefined);
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});
});

describe('Storage initialization with data and dtype = "float32"', () => {
	test('input: typeof data === "number"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(1, "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage([1, 2, 3], "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: typeof data === "boolean"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(true, "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage([true, false, true], "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Uint8Array && data.every((value) => value === 0 || value === 1); expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 1]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test("input: data instanceof Uint8Array && data.some((value) => value !== 0 && value !== 1); expect: this.dtype === 'float32', this.storage instanceof Float32Array", () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 2]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Float32Array; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Float32Array([1, 2, 3]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Float64Array; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Float64Array([1, 2, 3]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Int16Array; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Int16Array([1, 2, 3]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});

	test('input: data instanceof Int32Array; expect: this.dtype === "float32", this.storage instanceof Float32Array', () => {
		const storage = new Storage(new Int32Array([1, 2, 3]), "float32");
		expect(storage.dtype).toBe("float32");
		expect(storage.storage).toBeInstanceOf(Float32Array);
	});
});

describe('Storage initialization with data and dtype = "float64"', () => {
	test('input: typeof data === "number"; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(1, "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage([1, 2, 3], "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: typeof data === "boolean"; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(true, "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage([true, false, true], "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Uint8Array && data.every((value) => value === 0 || value === 1); expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 1]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test("input: data instanceof Uint8Array && data.some((value) => value !== 0 && value !== 1); expect: this.dtype === 'float64', this.storage instanceof Float64Array", () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 2]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Float32Array; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Float32Array([1, 2, 3]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Float64Array; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Float64Array([1, 2, 3]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Int16Array; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Int16Array([1, 2, 3]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});

	test('input: data instanceof Int32Array; expect: this.dtype === "float64", this.storage instanceof Float64Array', () => {
		const storage = new Storage(new Int32Array([1, 2, 3]), "float64");
		expect(storage.dtype).toBe("float64");
		expect(storage.storage).toBeInstanceOf(Float64Array);
	});
});

describe('Storage initialization with data and dtype = "int32"', () => {
	test('input: typeof data === "number"; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(1, "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage([1, 2, 3], "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: typeof data === "boolean"; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(true, "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage([true, false, true], "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: data instanceof Uint8Array && data.every((value) => value === 0 || value === 1); expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 1]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test("input: data instanceof Uint8Array && data.some((value) => value !== 0 && value !== 1); expect: this.dtype === 'int32', this.storage instanceof Int32Array", () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 2]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: data instanceof Float32Array; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Float32Array([1, 2, 3]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: data instanceof Float64Array; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Float64Array([1, 2, 3]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: data instanceof Int16Array; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Int16Array([1, 2, 3]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});

	test('input: data instanceof Int32Array; expect: this.dtype === "int32", this.storage instanceof Int32Array', () => {
		const storage = new Storage(new Int32Array([1, 2, 3]), "int32");
		expect(storage.dtype).toBe("int32");
		expect(storage.storage).toBeInstanceOf(Int32Array);
	});
});

describe('Storage initialization with data and dtype = "int16"', () => {
	test('input: typeof data === "number"; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(1, "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage([1, 2, 3], "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: typeof data === "boolean"; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(true, "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage([true, false, true], "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Uint8Array && data.every((value) => value === 0 || value === 1); expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 1]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test("input: data instanceof Uint8Array && data.some((value) => value !== 0 && value !== 1); expect: this.dtype === 'int16', this.storage instanceof Int16Array", () => {
		const storage = new Storage(new Uint8Array([0, 1, 0, 2]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Float32Array; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Float32Array([1, 2, 3]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Float64Array; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Float64Array([1, 2, 3]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Int16Array; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Int16Array([1, 2, 3]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});

	test('input: data instanceof Int32Array; expect: this.dtype === "int16", this.storage instanceof Int16Array', () => {
		const storage = new Storage(new Int32Array([1, 2, 3]), "int16");
		expect(storage.dtype).toBe("int16");
		expect(storage.storage).toBeInstanceOf(Int16Array);
	});
});

describe('Storage initialization with data and dtype = "bool"', () => {
	describe("data is a boolean or boolean[]", () => {
		test('input: typeof data === "boolean"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(true, "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: Array.isArray(data) && typeof data[0] === "boolean"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage([true, false, true], "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});
	});

	describe("data contains only 0s or 1s", () => {
		test('input: typeof data === "number"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(1, "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: Array.isArray(data) && typeof data[0] === "number"; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage([1, 0, 1], "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: data instanceof Uint8Array; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(new Uint8Array([0, 1, 0, 1]), "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: data instanceof Float32Array; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(new Float32Array([1, 0, 1]), "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: data instanceof Float64Array; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(new Float64Array([1, 0, 1]), "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: data instanceof Int16Array; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(new Int16Array([1, 0, 1]), "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});

		test('input: data instanceof Int32Array; expect: this.dtype === "bool", this.storage instanceof Uint8Array', () => {
			const storage = new Storage(new Int32Array([1, 0, 1]), "bool");
			expect(storage.dtype).toBe("bool");
			expect(storage.storage).toBeInstanceOf(Uint8Array);
		});
	});

	describe("data contains values other than 0s or 1s", () => {
		test('input: typeof data === "number"; expect: Error', () => {
			expect(() => new Storage(2, "bool")).toThrowError();
		});

		test('input: Array.isArray(data) && typeof data[0] === "number"; expect: Error', () => {
			expect(() => new Storage([1, 2, 1], "bool")).toThrowError();
		});

		test("input: data instanceof Uint8Array; expect: Error", () => {
			expect(
				() => new Storage(new Uint8Array([0, 2, 0, 1]), "bool")
			).toThrowError();
		});

		test("input: data instanceof Float32Array; expect: Error", () => {
			expect(
				() => new Storage(new Float32Array([1, 2, 1]), "bool")
			).toThrowError();
		});

		test("input: data instanceof Float64Array; expect: Error", () => {
			expect(
				() => new Storage(new Float64Array([1, 2, 1]), "bool")
			).toThrowError();
		});

		test("input: data instanceof Int16Array; expect: Error", () => {
			expect(
				() => new Storage(new Int16Array([1, 2, 1]), "bool")
			).toThrowError();
		});

		test("input: data instanceof Int32Array; expect: Error", () => {
			expect(
				() => new Storage(new Int32Array([1, 2, 1]), "bool")
			).toThrowError();
		});
	});
});
