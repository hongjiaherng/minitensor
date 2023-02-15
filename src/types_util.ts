/**
 * typing (not exposed to user's API)
 * - upcastType
 * - inferDTypeFromTensorLikeObj
 * - inferDTypeFromTypedArray
 * - castToTypedArray
 * - isDTypeMatchedWithTypedArray
 */
import { isTypedArray } from "util/types";
import { DType, TensorLike, TypedArray, TypedArrayMap } from "./types";

// Order of precedence of type upcasting: float64 > float32 > int32 > int16 > bool

const UpcastInt16AndMap = {
  [DType.float32]: DType.float32,
  [DType.float64]: DType.float64,
  [DType.int16]: DType.int16,
  [DType.int32]: DType.int32,
  [DType.bool]: DType.int16
};

const UpcastInt32AndMap = {
  [DType.float32]: DType.float32,
  [DType.float64]: DType.float64,
  [DType.int16]: DType.int32,
  [DType.int32]: DType.int32,
  [DType.bool]: DType.int32
};

const UpcastBoolAndMap = {
  [DType.float32]: DType.float32,
  [DType.float64]: DType.float64,
  [DType.int16]: DType.int16,
  [DType.int32]: DType.int32,
  [DType.bool]: DType.bool
};

const UpcastFloat64AndMap = {
  [DType.float32]: DType.float64,
  [DType.float64]: DType.float64,
  [DType.int16]: DType.float64,
  [DType.int32]: DType.float64,
  [DType.bool]: DType.float64
};

const UpcastFloat32AndMap = {
  [DType.float32]: DType.float32,
  [DType.float64]: DType.float64,
  [DType.int16]: DType.float32,
  [DType.int32]: DType.float32,
  [DType.bool]: DType.float32
};

const upcastTypeMap = {
  [DType.float32]: UpcastFloat32AndMap,
  [DType.float64]: UpcastFloat64AndMap,
  [DType.int16]: UpcastInt16AndMap,
  [DType.int32]: UpcastInt32AndMap,
  [DType.bool]: UpcastBoolAndMap
};

export function upcastType<D1 extends DType, D2 extends DType>(
  inputType: D1,
  otherType: D2
): D1 | D2 {
  return upcastTypeMap[inputType][otherType] as D1 | D2;
}

export function inferDTypeFromTensorLikeObj(tensorLikeObj: TensorLike): DType {
  if (
    typeof tensorLikeObj === "number" ||
    (Array.isArray(tensorLikeObj) && typeof tensorLikeObj[0] === "number")
  ) {
    return DType.float32;
  } else if (
    typeof tensorLikeObj === "boolean" ||
    (Array.isArray(tensorLikeObj) && typeof tensorLikeObj[0] === "boolean")
  ) {
    return DType.bool;
  } else if (isTypedArray(tensorLikeObj)) {
    return inferDTypeFromTypedArray(tensorLikeObj);
  }
  throw new Error("Cannot infer DType from TensorLike object");
}

export function inferDTypeFromTypedArray(data: TypedArray): DType {
  if (data instanceof Float64Array) {
    return DType.float64;
  } else if (data instanceof Float32Array) {
    return DType.float32;
  } else if (data instanceof Int16Array) {
    return DType.int16;
  } else if (data instanceof Int32Array) {
    return DType.int32;
  } else if (data instanceof Uint8Array) {
    if (data.some((d) => d !== 0 && d !== 1)) {
      throw new Error(
        "Uint8Array should contain only 0s and 1s to be casted to bool"
      );
    }
    return DType.bool;
  }
  throw new Error("Cannot infer DType from TypedArray object");
}

export function castToTypedArray<D extends DType>(
  tensorLikeObj: TensorLike,
  dtype: D
): TypedArrayMap[D] {
  let typedArray: TypedArray;

  if (
    isTypedArray(tensorLikeObj) &&
    isDTypeMatchedWithTypedArray(tensorLikeObj, dtype)
  ) {
    return tensorLikeObj as TypedArrayMap[D];
  }

  if (!Array.isArray(tensorLikeObj) && !isTypedArray(tensorLikeObj)) {
    tensorLikeObj = [tensorLikeObj] as number[] | boolean[];
  }

  switch (dtype) {
    case "float32":
      typedArray = Float32Array.from(tensorLikeObj as Array<number>);
      break;
    case "float64":
      typedArray = Float64Array.from(tensorLikeObj as Array<number>);
      break;
    case "int16":
      typedArray = Int16Array.from(tensorLikeObj as Array<number>);
      break;
    case "int32":
      typedArray = Int32Array.from(tensorLikeObj as Array<number>);
      break;
    case "bool":
      if (
        Array.isArray(tensorLikeObj) &&
        typeof tensorLikeObj[0] === "boolean"
      ) {
        typedArray = Uint8Array.from(tensorLikeObj as Array<number>);
      } else {
        if (
          (tensorLikeObj as TypedArray | Array<number>).some(
            (d) => d !== 0 && d !== 1
          )
        ) {
          throw new Error(
            "tensorLikeObj should contain only 0s and 1s to be casted to bool"
          );
        }
        typedArray = Uint8Array.from(tensorLikeObj as Array<number>);
      }
      break;

    default:
      throw new Error("Invalid dtype");
  }

  return typedArray as TypedArrayMap[D];
}

export function isDTypeMatchedWithTypedArray<D extends DType>(
  typedArray: TypedArray,
  dtype: D
): boolean {
  try {
    return inferDTypeFromTypedArray(typedArray) === dtype;
  } catch (error) {
    return false;
  }
}
