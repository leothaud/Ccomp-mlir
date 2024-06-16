//===--------------------------MyCastInterface.td--------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_INTERFACE
#define MYCAST_INTERFACE

include "mlir/IR/OpBase.td"

def MyCastInterface : OpInterface<"MyCastInterface"> {
  let methods = [
    InterfaceMethod<
      "Operations must have a prettyPrint method.",
      "void", "prettyPrint", (ins "::std::ostringstream &":$out, "int":$indent), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a checkDefinedVar method.",
      "llvm::SmallVector<std::string>", "checkDefinedVar", (ins "llvm::SmallVector<std::string>&":$declared), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a checkDefinedVar method.",
      "void", "checkDeclaredFun", (ins "llvm::SmallVector<std::string>&":$declared), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a addType method.",
      "mlir::LogicalResult", "addType", (ins "llvm::StringMap<mlir::Attribute> &":$globals,
          "llvm::StringMap<mlir::Attribute> &":$locals, "FunctionMap &":$funMap,
          "llvm::StringMap<llvm::StringMap<mlir::Attribute>> &":$fieldMap), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a getTypeAttr method.",
      "mlir::Attribute", "getTypeAttr", (ins ), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a isConst method.",
      "bool", "isConst", (ins ), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a isExtern method.",
      "bool", "isExtern", (ins ), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a isVolatile method.",
      "bool", "isVolatile", (ins ), [{}], [{ {} }]
    >,
    InterfaceMethod<
      "Operations must have a isStatic method.",
      "bool", "isStatic", (ins ), [{}], [{ {} }]
    >
  ];
  let cppNamespace = "::ccomp::myCast";
}

#endif // MYCAST_INTERFACE