//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef CAST_DIALECT
#define CAST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cast dialect definition.
//===----------------------------------------------------------------------===//

def Cast_Dialect : Dialect {
    let name = "cast";
    let summary = "A dialect representing an ast of a subset of C.";
    let description = [{
        This dialect represents the AST of the subset of the C language,
        adapted from standard version 17, parsed by the antlr grammar in
        this project.
    }];
    let cppNamespace = "::ccomp::cast";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Cast operation definition.
//===----------------------------------------------------------------------===//

class Cast_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cast_Dialect, mnemonic, traits>;

#endif // CAST_DIALECT