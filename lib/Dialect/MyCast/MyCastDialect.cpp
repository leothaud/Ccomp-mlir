//===---------------------MyCastDialect.cpp--------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//


#include "Dialect/MyCast/MyCastAttr.h"
#include "Dialect/MyCast/MyCastAttr.h.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MyCast/MyCastAttr.cpp.inc"
#undef GET_ATTRDEF_CLASSES

using namespace ccomp;
using namespace ccomp::myCast;

void MyCastDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/MyCast/MyCastOpsTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
  >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/MyCast/MyCastAttr.cpp.inc"
#undef GET_ATTRDEF_LIST
  >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/MyCast/MyCastOps.cpp.inc"
#undef GET_OP_LIST
      >();
}

#include "Dialect/MyCast/MyCastInterface.cpp.inc"