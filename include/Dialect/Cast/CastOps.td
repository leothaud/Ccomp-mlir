//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef CAST_OPS
#define CAST_OPS

include "CastDialect.td"
include "Dialect/Cast/CastOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"



def Cast_FooOp: Cast_Op<"foo"> {
    let arguments = (ins);
    let results = (outs Cast_ProgramNodeType:$res);
}

#endif // CAST_OPS