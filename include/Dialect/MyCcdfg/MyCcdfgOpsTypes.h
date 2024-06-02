//===--------------------------MyCcdfgOpsTypes.h---------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_TYPES_H__
#define MYCCDFG_TYPES_H__

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/MyCcdfg/MyCcdfgOpsTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

#endif // MYCCDFG_TYPES_H__