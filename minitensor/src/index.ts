export { Tensor } from "./tensor";
export { Storage } from "./storage";

export {
  arange,
  asStrided,
  empty,
  full,
  tensor,
  asStrided_,
  randNormal,
  randUniform
} from "./ops/creation";

export {
  reshape,
  slice,
  squeeze,
  reshape_,
  squeeze_,
  expand
} from "./ops/view";

export { add, add_, mul, mul_ } from "./ops/ewise_binary";

export {
  broadcastShapes,
  broadcastTensors,
  broadcastTo,
  isBroadcastedTensor,
  TensorsIterator,
  areBroadcastableTogether,
  isBroadcastableTo
} from "./ops/broadcast";

export { computeExpandedStrides, computeStrides } from "./shape_strides_util";
export { sharesMemory, isContiguous } from "./tensor_util";

import { DType } from "./types";
const float32 = DType.float32;
const float64 = DType.float64;
const int16 = DType.int16;
const int32 = DType.int32;
const bool = DType.bool;

export { float32, float64, int16, int32, bool, DType };

export * as datasets from "./datasets";
