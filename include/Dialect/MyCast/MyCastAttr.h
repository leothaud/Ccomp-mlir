//===--------------------MyCastAttr.h----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_ATTR_H__
#define MYCAST_ATTR_H__

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MyCast/MyCastAttr.h.inc"
#undef GET_ATTRDEF_CLASSES


#endif // MYCAST_ATTR_H__