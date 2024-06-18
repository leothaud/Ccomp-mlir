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
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base MyCast operation definition.
//===----------------------------------------------------------------------===//

class MyCast_UnpureOp<string mnemonic, list<Trait> traits = []> :
    Op<MyCast_Dialect, mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<MyCastInterface,
            ["prettyPrint", "checkDefinedVar", "checkDeclaredFun", "addType", "getTypeAttr",
            "isStatic", "isExtern", "isConst", "isVolatile", "isVolatile", "isConst",
            "isExtern", "isStatic", "variableUniquer"]>]
            )> {

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
        [{Op::checkDefinedVar(llvm::SmallVector<std::string> &declared) {
        }] # checkDefinedVarBody # [{
        }
    }];

    code checkDeclaredFunBody = [{
        for (auto op: getOperation()->getOperands())
            llvm::cast<MyCastInterface>(op.getDefiningOp()).checkDeclaredFun(declared);
    }];

    code checkDeclaredFunDefinition = [{
        void ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::checkDeclaredFun(llvm::SmallVector<std::string> &declared) {
        }] # checkDeclaredFunBody # [{
        }
    }];

    code addTypeBody = [{
        return mlir::success();
    }];

    code addTypeDefinition =  [{
        mlir::LogicalResult ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::addType(llvm::StringMap<mlir::Attribute> &globals, llvm::StringMap<mlir::Attribute> &locals,
                      FunctionMap &funMap, llvm::StringMap<llvm::StringMap<mlir::Attribute>> &fieldMap) {
        }] # addTypeBody # [{
        }
    }];

    code getTypeDefinition =  [{
        mlir::Attribute ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::getTypeAttr() {
            if (getOperation()->hasAttr("myCastType"))
                return getOperation()->getAttr("myCastType");
            return ccomp::myCast::UntypedAttr::get(getContext());
        }
    }];

    code isStatic = [{return false;}];
    code isConst = [{return false;}];
    code isExtern = [{return false;}];
    code isVolatile = [{return false;}];

    code isModifiedDefinition = [{
        bool ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::isStatic() {
        }] # isStatic # [{
        }
        bool ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::isConst() {
        }] # isConst # [{
        }
        bool ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::isExtern() {
        }] # isExtern # [{
        }
        bool ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::isVolatile() {
        }] # isVolatile # [{
        }
    }];

    code variableUniquerBody = [{
        for (auto op: getOperation()->getOperands())
            llvm::cast<MyCastInterface>(op.getDefiningOp()).variableUniquer(renamer, usedName);
    }];

    code variableUniquerDefinition = [{
        void ccomp::myCast::}] #
            !strconcat(!toupper(!substr(mnemonic,0,1)),
                !substr(mnemonic,1,!size(mnemonic))) #
        [{Op::variableUniquer(llvm::StringMap<std::string> &renamer, llvm::StringSet<> &usedName) {
        }] # variableUniquerBody # [{
        }
    }];

    let extraClassDefinition = variableUniquerDefinition #
                               isModifiedDefinition # getTypeDefinition #
                               addTypeDefinition # checkDeclaredFunDefinition #
                               checkDefinedVarDefinition # prettyPrintDefinition #
                               [{}];

}

class MyCast_Op<string mnemonic, list<Trait> traits = []> :
    MyCast_UnpureOp<mnemonic, !listconcat(traits, [Pure])> {}

#endif // MYCAST_DIALECT