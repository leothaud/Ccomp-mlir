//===-----------------------MyCcdfgOps.cpp----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#include "Dialect/MyCcdfg/MyCcdfgOps.h"
#include "Dialect/MyCcdfg/MyCcdfgDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/MyCcdfg/MyCcdfgOps.cpp.inc"