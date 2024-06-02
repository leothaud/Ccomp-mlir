//===---------------------MyCcdfgDialect.cpp--------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//


#include "Dialect/MyCcdfg/MyCcdfgDialect.h"
#include "Dialect/MyCcdfg/MyCcdfgOps.h"

using namespace ccomp;
using namespace ccomp::myCcdfg;

void MyCcdfgDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/MyCcdfg/MyCcdfgOpsTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
  >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/MyCcdfg/MyCcdfgOps.cpp.inc"
#undef GET_OP_LIST
      >();
}