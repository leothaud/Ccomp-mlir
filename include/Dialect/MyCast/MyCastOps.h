//===---------------------MyCastOps.h--------------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_CASTOPS_H
#define MYCAST_CASTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Dialect/MyCast/MyCastOpsTypes.h"

#define GET_OP_CLASSES
#include "Dialect/MyCast/MyCastOps.h.inc"

#endif // MYCAST_CASTOPS_H