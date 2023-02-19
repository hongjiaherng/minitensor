export { Tensor } from "./tensor";
export { Storage, sharesMemory } from "./storage";

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

export { add } from "./ops/ewise_binary";

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

import { DType } from "./types";
const float32 = DType.float32;
const float64 = DType.float64;
const int16 = DType.int16;
const int32 = DType.int32;
const bool = DType.bool;

export { float32, float64, int16, int32, bool };
