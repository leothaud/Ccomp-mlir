//===--------------------initAllDialect.h----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef CCOMP_INITALLDIALECT_H__
#define CCOMP_INITALLDIALECT_H__

#include "Dialect/MyCast/MyCastDialect.h"

namespace ccomp {

    inline void registerAllDialects(mlir::DialectRegistry &registry) {
        registry.insert<
            ccomp::myCast::MyCastDialect
        >();
    }

}

#endif // CCOMP_INITALLDIALECT_H__
