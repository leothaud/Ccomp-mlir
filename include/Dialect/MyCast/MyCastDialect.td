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
include "Dialect/MyCast/MyCastInterface.td"

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
    Op<MyCast_Dialect, mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<MyCastInterface, ["prettyPrint", "checkDefinedVar"]>])> {

    code prettyPrintBody = [{}];

    code prettyPrintDefinition = [{
        void ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::prettyPrint(std::ostringstream &out, int indent) {
        }] # prettyPrintBody # [{
        }
    }];

    code checkDefinedVarBody = [{
        for (auto op: getOperation()->getOperands())
            llvm::cast<MyCastInterface>(op.getDefiningOp()).checkDefinedVar(declared);
        return declared;
    }];

    code checkDefinedVarDefinition = [{
        llvm::SmallVector<std::string> ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::checkDefinedVar(llvm::SmallVector<std::string> declared) {
        }] # checkDefinedVarBody # [{
        }
    }];

    code typeCompatibleWithInt = [{false}];

    let extraClassDeclaration = [{
        static const bool compatibleWithInt = }] # typeCompatibleWithInt # [{;
    }];

    let extraClassDefinition = checkDefinedVarDefinition # prettyPrintDefinition # [{}];

}

#endif // MYCAST_DIALECT