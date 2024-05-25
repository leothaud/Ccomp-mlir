//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "Dialect/Cast/CastDialect.h"
#include "Dialect/Cast/CastOps.h"

using namespace ccomp;
using namespace ccomp::cast;

void CastDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Cast/CastOps.cpp.inc"
      >();
}