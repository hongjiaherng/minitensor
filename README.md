# Plan

- implement ndarray (the current tensor)
  - that supports basic operations (binaryOps, unaryOps, movementOps, reduceOps,
    conv)
  - that supports array broadcasting for binaryOps
  - reference to numjs for the structure and method to implement
- implement tensor
  - that utilizes ndarray internally
  - that supports autodiff

# How array broadcasting works

- given 2 vars, A & B
- compute targeted shape for broadcasting
  - broadcastShape(A, B) returns targeted shape & which variable to broadcast
- broadcast the smaller variable to the targeted shape
- compute the op

# Type Casting

# Ops

- viewOps: reshape, transpose, slice, concat, split
- unaryOps: abs, neg, ceil, floor, round, sqrt, exp, log, sin, cos, tan, asin,
  acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, sigmoid, tanh, relu,
  leakyRelu, elu, selu, softplus, softsign, logSoftmax, softmax, logSigmoid,
  logSoftmax, logSoftplus, logSoftsign, log1p, expm1, erf, erfc, sign, square,
  cube, reciprocal, rsqrt, square, cube, reciprocal, rsqrt, rcp, rsquare, rcube,
  exp2, log2, log10, log1p, expm1, erf, erfc, sign, square, cube, reciprocal,
  rsqrt, square, cube, reciprocal, rsqrt, rcp, rsquare, rcube, exp2, log2,
  log10, log1p, expm1, erf, erfc, sign, square, cube, reciprocal, rsqrt, square,
  cube, reciprocal, rsqrt, rcp, rsquare, rcube, exp2, log2, log10, log1p, expm1,
  erf, erfc, sign, square, cube, reciprocal, rsqrt, square, cube, reciprocal,
  rsqrt, rcp, rsquare, rcube, exp2, log2, log10, log1p, expm1, erf, erfc, sign,
  square, cube, reciprocal, rsqrt, square, cube, reciprocal, rsqrt, rcp,
  rsquare, rcube, exp2, log2, log10, log1p, expm1, erf, erfc, sign, square,
  cube, reciprocal, rsqrt, square, cube, reciprocal, rsqrt, rcp, rsquare, rcube,
  exp2, log2, log10, log1p, expm1, erf, erfc, sign, square, cube, reciprocal,
  rsqrt, square, cube, reciprocal, rsqrt, rcp, rsquare, rcube, exp2, log2,
  log10, log1p, expm1, erf, erfc, sign, square, cube, reciprocal, rsqrt, square,
  cube, reciprocal, rsqrt, rcp, rsquare, rcube, exp2, log2, log10,
- elementWiseBinaryOps: add, sub, mul, div, pow
- nonElementWiseBinaryOps: dot, cross, outer, matmul, conv
- reduceOps: sum, prod, mean, min, max, argmin, argmax, std, var, norm,
  logSumExp, logSum, logProd, logMean, logStd, logVar, logNorm, log1pSum,
  log1pProd, log1pMean, log1pStd, log1pVar, log1pNorm, expm1Sum, expm1Prod,
  expm1Mean, expm1Std, expm1Var, expm1Norm, erfSum, erfProd, erfMean, erfStd,
  erfVar, erfNorm, erfcSum, erfcProd, erfcMean, erfcStd, erfcVar, erfcNorm,
  signSum, signProd, signMean, signStd, signVar, signNorm, squareSum,
  squareProd, squareMean, squareStd, squareVar, squareNorm, cubeSum, cubeProd,
  cubeMean, cubeStd, cubeVar, cubeNorm, reciprocalSum, reciprocalProd,
  reciprocalMean, reciprocalStd, reciprocalVar, reciprocalNorm, rsqrtSum,
  rsqrtProd, rsqrtMean, rsqrtStd, rsqrtVar, rsqrtNorm, rcpSum, rcpProd, rcpMean,
  rcpStd, rcpVar, rcpNorm, rsquareSum, rsquareProd, rsquareMean, rsquareStd,
  rsquareVar, rsquareNorm, rcubeSum, rcubeProd, rcubeMean, rcubeStd, rcubeVar,
  rcubeNorm, exp2Sum, exp2Prod, exp2Mean, exp2Std, exp2Var, exp2Norm, log2Sum,
  log2Prod, log2Mean, log2Std, log2Var, log2Norm, log10Sum, log10Prod,
  log10Mean, log10Std, log10Var, log10Norm, log1pSum, log1pProd, log1pMean,
  log1pStd, log1pVar, log1pNorm, expm1Sum, expm1Prod, expm1Mean, expm1Std,
  expm1Var, expm1Norm, erfSum, erfProd, erfMean, erfStd, erfVar, erfNorm,
  erfcSum, erfcProd, erfcMean, erfcStd, erfcVar, erfcNorm, signSum, signProd,
  signMean, signStd,
- createOps: zeros, ones, eye, rand, randn, range, linspace, logspace, meshgrid,
  arange, repeat, tile, diag, diagflat, triu, tril, tri, vander, hstack, vstack,
  dstack, stack, columnStack, rowStack, block, meshgrid, arange, repeat, tile,
  diag, diagflat, triu, tril, tri, vander, hstack, vstack, dstack, stack,
  columnStack, rowStack, block, meshgrid, arange, repeat, tile, diag, diagflat,
  triu, tril, tri, vander, hstack, vstack, dstack, stack, columnStack, rowStack,
  block, meshgrid, arange, repeat, tile, diag, diagflat, triu, tril, tri,
  vander, hstack, vstack, dstack, stack, columnStack, rowStack, block, meshgrid,
  arange, repeat, tile, diag, diagflat, triu, tril, tri, vander, hstack, vstack,
  dstack, stack, columnStack, rowStack, block, meshgrid, arange, repeat, tile,
  diag, diagflat, triu, tril, tri, vander, hstack, vstack, dstack, stack,
  columnStack, rowStack, block, meshgrid, arange, repeat, tile, diag, diagflat,
  triu, tril, tri, vander, hstack, vstack, dstack, stack, columnStack, rowStack,
  block, meshgrid, arange, repeat, tile, diag, diagflat, triu, tril, tri,
  vander, hstack, vstack, dstack, stack, columnStack, rowStack, block, meshgrid,
  arange, repeat, tile, diag, diagflat, triu, tril, tri, vander, hstack, vstack,
  dstack, stack, columnStack, rowStack, block, meshgrid, arange, repeat, tile,
  diag, diagflat, triu, tril, tri, vander, hstack, vstack, dstack, stack,
  columnStack, rowStack, block, meshgrid, arange, repeat, tile, diag, diagflat,
  triu, tril, tri, vander, hstack, vstack
