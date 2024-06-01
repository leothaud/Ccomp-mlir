//===---------------------MyCastDialect.td---------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//


#ifndef MYCAST_DIALECT
#define MYCAST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MyCast dialect definition.
//===----------------------------------------------------------------------===//

def MyCast_Dialect : Dialect {
    let name = "myCast";
    let summary = "A dialect representing an ast of a language described in Ccomp/myc.bnf.";
    let description = [{
        This dialect represents the AST of the language described in Ccomp/myc.bnf.
        adapted from the C standard version 17, parsed by the antlr grammar in
        this project.
    }];
    let cppNamespace = "::ccomp::myCast";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base MyCast operation definition.
//===----------------------------------------------------------------------===//

class MyCast_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyCast_Dialect, mnemonic, traits>;

#endif // MyCAST_DIALECT