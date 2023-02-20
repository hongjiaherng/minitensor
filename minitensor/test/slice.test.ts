import * as minitensor from "../src";
import { describe, expect, test } from "@jest/globals";

// Testing with 2D tensor
const tensor = minitensor.reshape(minitensor.arange(1, 13), [4, 3]);

// Index a single element
describe("Single element indexing", () => {
  test("minitensor.slice(tensor, [0, 0])", () => {
    let sliced = minitensor.slice(tensor, [0, 0], true);
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0, 0], false);
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [0, { start: 0, stop: 1 }])", () => {
    let sliced = minitensor.slice(tensor, [0, { start: 0, stop: 1 }], true);
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0, { start: 0, stop: 1 }], false);
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 1 }, 0])", () => {
    let sliced = minitensor.slice(tensor, [{ start: 0, stop: 1 }, 0], true);
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [{ start: 0, stop: 1 }, 0], false);
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 1 }, { start: 0, stop: 1 }])", () => {
    let sliced = minitensor.slice(
      tensor,
      [
        { start: 0, stop: 1 },
        { start: 0, stop: 1 }
      ],
      true
    );
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [
        { start: 0, stop: 1 },
        { start: 0, stop: 1 }
      ],
      false
    );
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [-4, -3])", () => {
    let sliced = minitensor.slice(tensor, [-4, -3], true);
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [-4, -3], false);
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 1, step: 1 }, { start: 0, stop: 1, step: 1 }])", () => {
    let sliced = minitensor.slice(
      tensor,
      [
        { start: 0, stop: 1, step: 1 },
        { start: 0, stop: 1, step: 1 }
      ],
      true
    );
    expect(sliced.shape).toEqual([1, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [
        { start: 0, stop: 1, step: 1 },
        { start: 0, stop: 1, step: 1 }
      ],
      false
    );
    expect(sliced.shape).toEqual([1]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(sliced.size).toEqual(1);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });
});

// Slice a row
describe("Row slicing", () => {
  test("minitensor.slice(tensor, [0])", () => {
    let sliced = minitensor.slice(tensor, [0], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [0, null])", () => {
    let sliced = minitensor.slice(tensor, [0, null], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0, null], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 1 }, null])", () => {
    let sliced = minitensor.slice(tensor, [{ start: 0, stop: 1 }, null], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [{ start: 0, stop: 1 }, null], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 1, step: 1 }, null])", () => {
    let sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 1, step: 1 }, null],
      true
    );
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 1, step: 1 }, null],
      false
    );
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [-4, null])", () => {
    let sliced = minitensor.slice(tensor, [-4, null], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [-4, null], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: -4, stop: -3, step: 1 }, null])", () => {
    let sliced = minitensor.slice(
      tensor,
      [{ start: -4, stop: -3, step: 1 }, null],
      true
    );
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [{ start: -4, stop: -3, step: 1 }, null],
      false
    );
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [0, { start: 0, stop: 3 }])", () => {
    let sliced = minitensor.slice(tensor, [0, { start: 0, stop: 3 }], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0, { start: 0, stop: 3 }], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [0, { start: 0, stop: 3, step: 1 }])", () => {
    let sliced = minitensor.slice(
      tensor,
      [0, { start: 0, stop: 3, step: 1 }],
      true
    );
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [0, { start: 0, stop: 3, step: 1 }],
      false
    );
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [0, { start: -3, stop: 3 }])", () => {
    let sliced = minitensor.slice(tensor, [0, { start: -3, stop: 3 }], true);
    expect(sliced.shape).toEqual([1, 3]);
    expect(sliced.strides).toEqual([3, 1]);
    console.log(sliced.offset);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [0, { start: -3, stop: 3 }], false);
    expect(sliced.shape).toEqual([3]);
    expect(sliced.strides).toEqual([1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });
});

// Slice a column
describe("Column slicing", () => {
  test("minitensor.slice(tensor, [null, 0])", () => {
    let sliced = minitensor.slice(tensor, [null, 0], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [null, 0], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [null, { start: 0, stop: 1 }])", () => {
    let sliced = minitensor.slice(tensor, [null, { start: 0, stop: 1 }], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [null, { start: 0, stop: 1 }], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [null, { start: 0, stop: 1, step: 1 }])", () => {
    let sliced = minitensor.slice(
      tensor,
      [null, { start: 0, stop: 1, step: 1 }],
      true
    );
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [null, { start: 0, stop: 1, step: 1 }],
      false
    );
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [null, -3])", () => {
    let sliced = minitensor.slice(tensor, [null, -3], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [null, -3], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [null, { start: -3, stop: 1 }])", () => {
    let sliced = minitensor.slice(tensor, [null, { start: -3, stop: 1 }], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [null, { start: -3, stop: 1 }], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [null, { start: -3, stop: -2 }])", () => {
    let sliced = minitensor.slice(
      tensor,
      [null, { start: -3, stop: -2 }],
      true
    );
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [null, { start: -3, stop: -2 }], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4 }, 0])", () => {
    let sliced = minitensor.slice(tensor, [{ start: 0, stop: 4 }, 0], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [{ start: 0, stop: 4 }, 0], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: 1 }, 0])", () => {
    let sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 4, step: 1 }, 0],
      true
    );
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 4, step: 1 }, 0],
      false
    );
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: -4, stop: 4 }, 0])", () => {
    let sliced = minitensor.slice(tensor, [{ start: -4, stop: 4 }, 0], true);
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(tensor, [{ start: -4, stop: 4 }, 0], false);
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([3]);
    expect(sliced.offset).toEqual(0);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });
});

// Slice a sub-tensor with shape
describe("Sub-tensor slicing", () => {
  test("minitensor.slice(tensor, [{ start: 1, stop: 3 }, { start: 0, stop: 2 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: 1, stop: 3 },
      { start: 0, stop: 2 }
    ]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(3);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: -3, stop: -1 }, { start: 0, stop: 2 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: -3, stop: -1 },
      { start: 0, stop: 2 }
    ]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(3);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 1, stop: 3 }, { start: -3, stop: -1 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: 1, stop: 3 },
      { start: -3, stop: -1 }
    ]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(3);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: -3, stop: -1 }, { start: -3, stop: -1 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: -3, stop: -1 },
      { start: -3, stop: -1 }
    ]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(sliced.strides).toEqual([3, 1]);
    expect(sliced.offset).toEqual(3);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });
});

// Slice with negative step
describe("Negative step slicing", () => {
  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: -1 }, 0])", () => {
    let sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 4, step: -1 }, 0],
      true
    );
    expect(sliced.shape).toEqual([4, 1]);
    expect(sliced.strides).toEqual([-3, 1]);
    expect(sliced.offset).toEqual(9);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);

    sliced = minitensor.slice(
      tensor,
      [{ start: 0, stop: 4, step: -1 }, 0],
      false
    );
    expect(sliced.shape).toEqual([4]);
    expect(sliced.strides).toEqual([-3]);
    expect(sliced.offset).toEqual(9);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: -1 }, { start: 0, stop: 2, step: -1 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: 0, stop: 4, step: -1 },
      { start: 0, stop: 2, step: -1 }
    ]);
    expect(sliced.shape).toEqual([4, 2]);
    expect(sliced.strides).toEqual([-3, -1]);
    expect(sliced.offset).toEqual(10);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: -2 }, { start: 0, stop: 2 }])", () => {
    let sliced = minitensor.slice(tensor, [
      { start: 0, stop: 4, step: -2 },
      { start: 0, stop: 2 }
    ]);
    expect(sliced.shape).toEqual([2, 2]);
    expect(sliced.strides).toEqual([-6, 1]);
    expect(sliced.offset).toEqual(9);
    expect(minitensor.sharesMemory(sliced, tensor)).toBe(true);
  });
});

// Try to slice with invalid arguments
describe("Invalid slicing", () => {
  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: 0 }, 0])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [
        { start: 0, stop: 4, step: 0 },
        0
      ]);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: 0 }, { start: 0, stop: 2, step: 0 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [
        { start: 0, stop: 4, step: 0 },
        { start: 0, stop: 2, step: 0 }
      ]);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4, step: 0 }, { start: 0, stop: 2 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [
        { start: 0, stop: 4, step: 0 },
        { start: 0, stop: 2 }
      ]);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4 }, { start: 0, stop: 2, step: 0 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [
        { start: 0, stop: 4 },
        { start: 0, stop: 2, step: 0 }
      ]);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, []);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [{ start: 0, stop: 4 }, { start: 0, stop: 2 }, { start: 0, stop: 2 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [
        { start: 0, stop: 4 },
        { start: 0, stop: 2 },
        { start: 0, stop: 2 }
      ]);
    }).toThrowError();
  });

  test("minitensor.slice(tensor, [0, 0, 0])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [0, 0, 0]);
    }).toThrowError();
  })

  test("minitensor.slice(tensor, [{ start: 4 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [{ start: 4 }]);
    }).toThrowError();
  })

  test("minitensor.slice(tensor, [{ start: 4, stop: 0 }])", () => {
    expect(() => {
      let sliced = minitensor.slice(tensor, [{ start: 4, stop: 0 }]);
    }).toThrowError();
  })  

});
