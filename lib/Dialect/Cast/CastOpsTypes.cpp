//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/DialectImplementation.h"

#include "Dialect/Cast/CastDialect.h"
#include "Dialect/Cast/CastOpsTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Cast/CastOpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES