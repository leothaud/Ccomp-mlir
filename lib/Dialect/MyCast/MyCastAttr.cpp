//===-----------------MyCastAttr.cpp-----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/DialectImplementation.h"

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastAttr.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MyCast/MyCastAttr.cpp.inc"
#undef GET_ATTRDEF_CLASSES