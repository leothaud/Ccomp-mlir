//===----------------------------MyCcdfgOps.td-----------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_OPS
#define MYCCDFG_OPS

include "Dialect/MyCcdfg/MyCcdfgDialect.td"
include "Dialect/MyCcdfg/MyCcdfgOpsTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

def MyCcdfg_FooOp : MyCcdfg_Op<"foo"> {
    let arguments = (ins );
    let results = (outs I32:$res);
}

#endif // MYCCDFG_OPS