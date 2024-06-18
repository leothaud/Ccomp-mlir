//===---------------------MyCastDialect.h----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_MYCASTDIALECT_H
#define MYCAST_MYCASTDIALECT_H

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSet.h"

struct FunctionMap {
    llvm::StringMap<mlir::Attribute> returnType;
    llvm::StringMap<llvm::SmallVector<mlir::Attribute>> arguments;

    bool contains(std::string &name) {
        return returnType.contains(name);
    }

    mlir::Attribute &getReturnType(std::string &name) {
        return returnType[name];
    }

    llvm::SmallVector<mlir::Attribute> &getArguments(std::string &name) {
        return arguments[name];
    }

    void addFunction(std::string &name, mlir::Attribute retType) {
        returnType[name] = retType;
        arguments[name] = llvm::SmallVector<mlir::Attribute>();
    }

    void addArgument(std::string &name, mlir::Attribute argType) {
        arguments[name].push_back(argType);
    }
};

#include "Dialect/MyCast/MyCastOpsDialect.h.inc"
#include "Dialect/MyCast/MyCastAttr.h.inc"
#include "Dialect/MyCast/MyCastInterface.h.inc"

#endif // MYCAST_MYCASTDIALECT_H