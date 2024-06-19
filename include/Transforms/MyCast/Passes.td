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

def TypeAstPass : Pass<"type-ast", "ccomp::myCast::ProgramOp"> {
  let summary = "add a type attribute to every AST node.";
  let constructor = "ccomp::myCast::createTypeAstPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def ReplaceGenericPass : Pass<"replace-generic", "ccomp::myCast::ProgramOp"> {
  let summary = "replace generic expression with corresponding expression.";
  let constructor = "ccomp::myCast::createReplaceGenericPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def CompleteTypeAstPass : Pass<"complete-type-ast", "ccomp::myCast::ProgramOp"> {
  let summary = "type myC ast and apply useful pass.";
  let constructor = "ccomp::myCast::createCompleteTypeAstPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def VariableUniquerPass : Pass<"variable-uniquer", "ccomp::myCast::ProgramOp"> {
  let summary = "make myC variables' name unique.";
  let constructor = "ccomp::myCast::createVariableUniquerPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def ForAndDoToWhilePass : Pass<"forNdo-to-while", "ccomp::myCast::ProgramOp"> {
  let summary = "replace for and do-while loops to while loops.";
  let constructor = "ccomp::myCast::createForAndDoToWhilePass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def FlattenCompoundPass : Pass<"flatten-compound", "ccomp::myCast::ProgramOp"> {
  let summary = "replace for and do-while loops to while loops.";
  let constructor = "ccomp::myCast::createFlattenCompoundPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def PullUpVarDeclsPass : Pass<"pull-up-var-decls", "ccomp::myCast::ProgramOp"> {
  let summary = "put every var declaration statement at the top of the function body.";
  let constructor = "ccomp::myCast::createPullUpVarDeclsPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

def PropagateConstPass : Pass<"propagate-const", "ccomp::myCast::ProgramOp"> {
  let summary = "propagate const variable values.";
  let constructor = "ccomp::myCast::createPropagateConstPass()";
  let dependentDialects = ["ccomp::myCast::MyCastDialect"];
}

#endif // MYCAST_PASSES_TD