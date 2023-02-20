import * as minitensor from "../src";
import { describe, expect, test } from "@jest/globals";

describe("Adding 2 tensors", () => {
  test("add", () => {
    const a = minitensor.tensor([1, 2, 3]);
    const b = minitensor.tensor([4, 5, 6]);
    const c = minitensor.add(a, b);
    expect(c.array()).toEqual([5, 7, 9]);
  });
});

const fixedShape = [4];
const dataFloat64 = [1.123, 2.123, 3.123, 4.123];
const dataFloat32 = [1.123, 2.123, 3.123, 4.123];
const dataInt32 = [1, 2, 3, 4];
const dataInt16 = [1, 2, 3, 4];
const dataBool = [true, false, true, false];

const resultFloat64Float64 = [2.246, 4.246, 6.246, 8.246];
const resultFloat64Float32 = [2.246, 4.246, 6.246, 8.246];
const resultFloat64Int32 = [2.123, 4.123, 6.123, 8.123];
const resultFloat64Int16 = [2.123, 4.123, 6.123, 8.123];
const resultFloat64Bool = [2.123, 2.123, 4.123, 4.123];

const resultFloat32Float32 = [2.246, 4.246, 6.246, 8.246];
const resultFloat32Int32 = [2.123, 4.123, 6.123, 8.123];
const resultFloat32Int16 = [2.123, 4.123, 6.123, 8.123];
const resultFloat32Bool = [2.123, 2.123, 4.123, 4.123];

const resultInt32Int32 = [2, 4, 6, 8];
const resultInt32Int16 = [2, 4, 6, 8];
const resultInt32Bool = [2, 2, 4, 4];

const resultInt16Int16 = [2, 4, 6, 8];
const resultInt16Bool = [2, 2, 4, 4];

const resultBoolBool = [true, false, true, false];

// Non-inplace typecasting test
describe("Typecasting for copied / non-in-place op", () => {
  describe("Typecasting for copied / non-in-place op (float64)", () => {
    test("input: float64, other: float64 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const b = minitensor.tensor(dataFloat64, fixedShape, minitensor.float32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float64);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Float64[i], 3);
      });
    });

    test("input: float64, other: float32 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float64);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Float32[i], 3);
      });
    });

    test("input: float64, other: int32 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float64);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Int32[i], 3);
      });
    });

    test("input: float64, other: int16 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float64);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Int16[i], 3);
      });
    });

    test("input: float64, other: bool => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float64);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Bool[i], 3);
      });
    });
  });

  describe("Typecasting for copied / non-in-place op (float32)", () => {
    test("input: float32, other: float32 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float32);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Float32[i], 3);
      });
    });

    test("input: float32, other: int32 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float32);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Int32[i], 3);
      });
    });

    test("input: float32, other: int16 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float32);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Int16[i], 3);
      });
    });

    test("input: float32, other: bool => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.float32);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Bool[i], 3);
      });
    });
  });

  describe("Typecasting for copied / non-in-place op (int32)", () => {
    test("input: int32, other: int32 => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.int32);
      expect(c.array()).toEqual(resultInt32Int32);
    });

    test("input: int32, other: int16 => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.int32);
      expect(c.array()).toEqual(resultInt32Int16);
    });

    test("input: int32, other: bool => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.int32);
      expect(c.array()).toEqual(resultInt32Bool);
    });
  });

  describe("Typecasting for copied / non-in-place op (int16)", () => {
    test("input: int16, other: int16 => result: int16", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.int16);
      expect(c.array()).toEqual(resultInt16Int16);
    });

    test("input: int16, other: bool => result: int16", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.int16);
      expect(c.array()).toEqual(resultInt16Bool);
    });
  });

  describe("Typecasting for copied / non-in-place op (bool)", () => {
    test("input: bool, other: bool => result: bool", () => {
      const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add(a, b);
      expect(c.dtype).toEqual(minitensor.bool);
      expect(c.array()).toEqual(resultBoolBool);
    });
  });
});

describe("Typecasting for in-place op", () => {
  describe("Typecasting for in-place op (float64)", () => {
    test("input: float64, other: float64 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Float64[i], 3);
      });
    });

    test("input: float64, other: float32 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Float32[i], 3);
      });
    });

    test("input: float64, other: int32 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Int32[i], 3);
      });
    });

    test("input: float64, other: int16 => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Int16[i], 3);
      });
    });

    test("input: float64, other: bool => result: float64", () => {
      const a = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Bool[i], 3);
      });
    });
  });

  describe("Typecasting for in-place op (float32)", () => {
    test("input: float32, other: float64 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat64Float32[i], 3);
      });
    });

    test("input: float32, other: float32 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Float32[i], 3);
      });
    });

    test("input: float32, other: int32 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Int32[i], 3);
      });
    });

    test("input: float32, other: int16 => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Int16[i], 3);
      });
    });

    test("input: float32, other: bool => result: float32", () => {
      const a = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);

      c.array().forEach((v, i) => {
        expect(v).toBeCloseTo(resultFloat32Bool[i], 3);
      });
    });
  });

  describe("Typecasting for in-place op (int32)", () => {
    test("input: int32, other: float64 => result: float64", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int32, other: float32 => result: float32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int32, other: int32 => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int32, other: int16 => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int32, other: bool => result: int32", () => {
      const a = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 2, 4, 4]);
    });
  });

  describe("Typecasting for in-place op (int16)", () => {
    test("input: int16, other: float64 => result: float64", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat64, fixedShape, minitensor.float64);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int16, other: float32 => result: float32", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataFloat32, fixedShape, minitensor.float32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int16, other: int32 => result: int32", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt32, fixedShape, minitensor.int32);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int16, other: int16 => result: int16", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 4, 6, 8]);
    });

    test("input: int16, other: bool => result: int16", () => {
      const a = minitensor.tensor(dataInt16, fixedShape, minitensor.int16);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([2, 2, 4, 4]);
    });
  });

  describe("Typecasting for in-place op (bool)", () => {
    test("input: bool, other: float64 => result: float64", () => {
      expect(() => {
        const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
        const b = minitensor.tensor(
          dataFloat64,
          fixedShape,
          minitensor.float64
        );
        const c = minitensor.add_(a, b);
      }).toThrowError();
    });

    test("input: bool, other: float32 => result: float32", () => {
      expect(() => {
        const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
        const b = minitensor.tensor(
          dataFloat32,
          fixedShape,
          minitensor.float32
        );
        const c = minitensor.add_(a, b);
      }).toThrowError();
    });

    test("input: bool, other: int32 => result: int32", () => {
      expect(() => {
        const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
        const b = minitensor.tensor(
          dataInt32,
          fixedShape,
          minitensor.int32
        );
        const c = minitensor.add_(a, b);
      }).toThrowError();
    });

    test("input: bool, other: int16 => result: int16", () => {
      expect(() => {
        const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
        const b = minitensor.tensor(
          dataInt16,
          fixedShape,
          minitensor.int16
        );
        const c = minitensor.add_(a, b);
      }).toThrowError();
    });

    test("input: bool, other: bool => result: bool", () => {
      const a = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const aCopied = a.clone();
      const b = minitensor.tensor(dataBool, fixedShape, minitensor.bool);
      const c = minitensor.add_(a, b);
      expect(minitensor.sharesMemory(a, c)).toBe(true);
      expect(minitensor.sharesMemory(c, aCopied)).toBe(false);
      expect(c.array()).toEqual([true, false, true, false]);
    });
  });
});
