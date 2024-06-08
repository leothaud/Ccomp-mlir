//===---------------------------initAllPasses.h----------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef CCOMP_INITALLPASSES_H__
#define CCOMP_INITALLPASSES_H__

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCcdfg/MyCcdfgDialect.h"
#include "Dialect/MyCcdfg/MyCcdfgOps.h"
#include "Transforms/MyCast/Passes.h"

namespace ccomp {
inline void registerAllPasses() {

  static bool initOnce = []() {
    myCast::registerTransformsPasses();
    return true;
  }();
  (void)initOnce;
}

} // namespace ccomp

#endif // CCOMP_INITALLPASSES_H__