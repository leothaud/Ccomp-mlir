//===------------------------------Passes.td-------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
#ifndef MYCAST_PASSES_TD
#define MYCAST_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"
include "Dialect/MyCast/MyCastDialect.td"
include "Dialect/MyCast/MyCastOps.td"

def PrettyPrintAstPass : Pass<"pretty-print-ast", "ccomp::myCast::ProgramOp"> {
  let summary = "pretty print MyC ast.";
  let constructor = "ccomp::myCast::createPrettyPrintAstPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
  let options = [
    Option<"filename", "filename", "::std::string", "\"\"",
           "file in which export prettyprinted ast.">
  ];
}

#endif // MYCAST_PASSES_TD