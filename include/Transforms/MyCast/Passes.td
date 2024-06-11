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

def CheckDefinedVarPass : Pass<"check-defined-var", "ccomp::myCast::ProgramOp"> {
  let summary = "check if every used variable has been declared.";
  let constructor = "ccomp::myCast::createCheckDefinedVarPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def CheckDeclaredFunPass : Pass<"check-declared-fun", "ccomp::myCast::ProgramOp"> {
  let summary = "check if every used function has been declared.";
  let constructor = "ccomp::myCast::createCheckDeclaredFunPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def ReplaceAliasPass : Pass<"replace-alias", "ccomp::myCast::ProgramOp"> {
  let summary = "replace every alias with corresponding type.";
  let constructor = "ccomp::myCast::createReplaceAliasPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

#endif // MYCAST_PASSES_TD