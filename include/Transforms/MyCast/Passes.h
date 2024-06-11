//===-------------------------------Passes.h-------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_PASSES
#define MYCAST_PASSES

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace ccomp {
    namespace myCast {

        std::unique_ptr<mlir::OperationPass<ccomp::myCast::ProgramOp>> createPrettyPrintAstPass();
        std::unique_ptr<mlir::OperationPass<ccomp::myCast::ProgramOp>> createCheckDefinedVarPass();
        std::unique_ptr<mlir::OperationPass<ccomp::myCast::ProgramOp>> createCheckDeclaredFunPass();
        std::unique_ptr<mlir::OperationPass<ccomp::myCast::ProgramOp>> createReplaceAliasPass();


#define GEN_PASS_DECL_PRETTYPRINTASTPASS
#define GEN_PASS_DEF_PRETTYPRINTASTPASS
#define GEN_PASS_DECL_CHECKDEFINEDVARPASS
#define GEN_PASS_DEF_CHECKDEFINEDVARPASS
#define GEN_PASS_DECL_CHECKDECLAREDFUNPASS
#define GEN_PASS_DEF_CHECKDECLAREDFUNPASS
#define GEN_PASS_DECL_REPLACEALIASPASS
#define GEN_PASS_DEF_REPLACEALIASPASS
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
        #include "Transforms/MyCast/Passes.h.inc"

    } // namespace myCast
} // namespace ccomp

#endif // MYCAST_PASSES