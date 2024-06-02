//===--------------------------MyCcdfgDialect.td---------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_DIALECT
#define MYCCDFG_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MyCcdfg dialect definition.
//===----------------------------------------------------------------------===//

def MyCcdfg_Dialect : Dialect {
    let name = "myCcdfg";
    let summary = "A dialect representing an cdfg of a language described in Ccomp/myc.bnf.";
    let description = [{
        This dialect represents the cdfg of the language described in Ccomp/myc.bnf.
        adapted from the C standard version 17, parsed by the antlr grammar in
        this project.
    }];
    let cppNamespace = "::ccomp::myCcdfg";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base MyCcdfg operation definition.
//===----------------------------------------------------------------------===//

class MyCcdfg_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyCcdfg_Dialect, mnemonic, traits>;

#endif // MYCCDFG_DIALECT