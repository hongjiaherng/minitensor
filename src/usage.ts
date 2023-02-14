import * as minitensor from ".";

const a = minitensor.tensor([1, 2, 3, 4], [2, 2]);
console.log(minitensor.expand(a, [2, 2, -1]))


// const a = minitensor.tensor([[[1, 2, 3]], [[4, 5, 6]]]);
// console.log(a.array());
// console.log(a);

// const b = minitensor.tensor([1, 2, 3, 4], [2, 2]);
// console.log(b.array());
// console.log(b);

// const c = minitensor.tensor([1, 2, 3, 4]);
// console.log(c.array());
// console.log(c);

// const d = minitensor.tensor([[1, 2], [3, 4]], [1, 4]);
// console.log(d.array());
// console.log(d);

// const a = minitensor.tensor([[[1, 2, 3]], [[4, 5, 6]]]);
// console.log(a.array());
// console.log("a.shape", a.shape);
// console.log("a.strides", a.strides)

// const b = expand(a, [2, 3, 3])
// // b.array().forEach((x) => console.log(x));
// console.log(b.array());
// console.log("b.shape", b.shape);
// console.log("b.strides", b.strides)



// console.log(broadcastTensors(minitensor.tensor([1, 2, 3]), minitensor.tensor([2, 2, 3])));


// const targetShape = [2, 3]

// const input1 = minitensor.tensor([[100], [200]])
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input1, [1]).array(), "\n");

// console.log("input", input1.array());
// console.log("input shape", input1.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input1, targetShape).array(), "\n");

// const input2 = minitensor.tensor([100, 200, 300])
// console.log("input", input2.array());
// console.log("input shape", input2.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input2, targetShape).array(), "\n");

// const input3 = minitensor.tensor([[100, 200, 300]])
// console.log("input", input3.array());
// console.log("input shape", input3.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input3, targetShape).array(), "\n");

// // TODO: number is not supported
// // const input4 = 100
// // console.log("input", input4);
// // console.log("input shape", "scalar");
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input4, targetShape).array(), "\n");

// const input5 = minitensor.tensor([100])
// console.log("input", input5.array());
// console.log("input shape", input5.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input5, targetShape).array(), "\n");

// const input6 = minitensor.tensor([[100]])
// console.log("input", input6.array());
// console.log("input shape", input6.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input6, targetShape).array(), "\n");

// // TODO: hang
// const input7 = minitensor.tensor([[[100]]])
// // console.log("input", input7.array());
// // console.log("input shape", input7.shape);
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input7, targetShape).array(), "\n");

// // Error expected
// // const input8 = minitensor.tensor([100, 200])
// // console.log("input", input8.array());
// // console.log("input shape", input8.shape);
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input8, targetShape).array(), "\n");

// // Error expected
// // const input8 = minitensor.tensor([[100, 200]])
// // console.log("input", input8.array());
// // console.log("input shape", input8.shape);
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input8, targetShape).array(), "\n");

// // Error expected
// // const input9 = minitensor.tensor([[100], [200], [300]])
// // console.log("input", input9.array());
// // console.log("input shape", input9.shape);
// // console.log("broadcasted", minitensor.broadcast.broadcastTo(input9, targetShape).array(), "\n");

// const input10 = minitensor.tensor([[100, 200, 300], [400, 500, 600]]);
// console.log("input", input10.array());
// console.log("input shape", input10.shape);
// console.log("broadcasted", minitensor.broadcast.broadcastTo(input10, targetShape).array(), "\n");






