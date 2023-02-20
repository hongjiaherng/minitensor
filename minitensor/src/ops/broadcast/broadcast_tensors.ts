import { areShapesEqual } from "../../shape_strides_util";
import { Tensor } from "../../tensor";
import { DType } from "../../types";
import { broadcastShapes } from "./broadcast_shapes";
import { _broadcastTo } from "./broadcast_to";

export function broadcastTensors(...tensors: Tensor<any>[]): Tensor<any>[] {
  if (tensors.length < 2) return tensors;
  if (areShapesEqual(...tensors.map((t) => t.shape))) return tensors;

  const targetShape = broadcastShapes(...tensors.map((t) => t.shape));
  return tensors.map((t) => _broadcastTo(t, targetShape));
}