import { broadcastShapes } from "./ops/broadcast/broadcast_shapes";
import { broadcastTo } from "./ops/broadcast/broadcast_to";
import { TensorsIterator } from "./ops/broadcast/tensors_iterator";
import { empty } from "./ops/creation/empty";
import { full } from "./ops/creation/full";
import { tensor } from "./ops/creation/tensor";
import { add } from "./ops/ewise_binary/add";
import { inferUnknownDimension, reshape } from "./ops/view/reshape";
import { Storage } from "./storage";
// import { Tensor } from "./tensor";
import {
	castToTypedArray,
	inferDTypeFromTensorLikeObj,
	TensorLike,
	upcastType,
} from "./types";
import { flattenArray, inferShape } from "./utils";

const a = tensor([100, 200, 300], "int32");
const b = reshape(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], "float32"), [3, 3]);

add(a, b);
