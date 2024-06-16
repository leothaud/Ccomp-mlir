//===--------------------MyCastAttr.h----------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_ATTR_H__
#define MYCAST_ATTR_H__

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/MyCast/MyCastAttr.h.inc"
#undef GET_ATTRDEF_CLASSES

static inline bool canBeCond(mlir::Attribute attr) {
    if (auto intAttr = llvm::dyn_cast<ccomp::myCast::IntegerAttr>(attr))
        return true;
    return false;

}

static inline bool areCompatible(mlir::Attribute attr1, mlir::Attribute attr2) {
    if (auto intAttr1 = llvm::dyn_cast<ccomp::myCast::IntegerAttr>(attr1)) {
        if (auto intAttr2 = llvm::dyn_cast<ccomp::myCast::IntegerAttr>(attr2)) 
            return true;
        return false;

    }
    if (auto floatAttr1 = llvm::dyn_cast<ccomp::myCast::FloatAttr>(attr1)) {
        if (auto floatAttr2 = llvm::dyn_cast<ccomp::myCast::FloatAttr>(attr2))
            return true;
        return false;
    }
    if (auto ptrAttr1 = llvm::dyn_cast<ccomp::myCast::PtrAttr>(attr1)) {
        if (auto ptrAttr2 = llvm::dyn_cast<ccomp::myCast::PtrAttr>(attr2))
            return areCompatible(ptrAttr1.getBaseType(), ptrAttr2.getBaseType());
        return false;
    }
    if (auto enumAttr1 = llvm::dyn_cast<ccomp::myCast::EnumAttr>(attr1)) {
        if (auto enumAttr2 = llvm::dyn_cast<ccomp::myCast::EnumAttr>(attr2)) {
            std::string enumName = enumAttr1.getName().getValue().data();
            return enumName.compare(enumAttr2.getName().getValue().data()) == 0;
        }
        return false;
    }
    if (auto unionAttr1 = llvm::dyn_cast<ccomp::myCast::UnionAttr>(attr1)) {
        if (auto unionAttr2 = llvm::dyn_cast<ccomp::myCast::UnionAttr>(attr2)) {
            std::string unionName = unionAttr1.getName().getValue().data();
            return unionName.compare(unionAttr2.getName().getValue().data()) == 0;
        }
        return false;
    }
    if (auto structAttr1 = llvm::dyn_cast<ccomp::myCast::StructAttr>(attr1)) {
        if (auto structAttr2 = llvm::dyn_cast<ccomp::myCast::StructAttr>(attr2)) {
            std::string structName = structAttr1.getName().getValue().data();
            return structName.compare(structAttr2.getName().getValue().data()) == 0;
        }
        return false;

    }
    if (auto voidAttr1 = llvm::dyn_cast<ccomp::myCast::VoidAttr>(attr1)) {
        if (auto voidAttr2 = llvm::dyn_cast<ccomp::myCast::VoidAttr>(attr2))
            return true;
        return false;
    }
    return false;
}

static inline bool areEquals(mlir::Attribute attr1, mlir::Attribute attr2) {
    if (auto intAttr1 = llvm::dyn_cast<ccomp::myCast::IntegerAttr>(attr1)) {
        if (auto intAttr2 = llvm::dyn_cast<ccomp::myCast::IntegerAttr>(attr2)) 
            return (intAttr1.getIsSigned() == intAttr2.getIsSigned()) && (intAttr1.getBw() == intAttr2.getBw());
        return false;

    }
    if (auto floatAttr1 = llvm::dyn_cast<ccomp::myCast::FloatAttr>(attr1)) {
        if (auto floatAttr2 = llvm::dyn_cast<ccomp::myCast::FloatAttr>(attr2))
            return (floatAttr1.getIsSigned() == floatAttr2.getIsSigned()) &&
                    (floatAttr1.getEbw() == floatAttr2.getEbw()) &&
                    (floatAttr1.getMbw() == floatAttr2.getMbw());
        return false;
    }
    if (auto ptrAttr1 = llvm::dyn_cast<ccomp::myCast::PtrAttr>(attr1)) {
        if (auto ptrAttr2 = llvm::dyn_cast<ccomp::myCast::PtrAttr>(attr2))
            return areEquals(ptrAttr1.getBaseType(), ptrAttr2.getBaseType()) && (ptrAttr1.getSize() == ptrAttr2.getSize());
        return false;
    }
    if (auto enumAttr1 = llvm::dyn_cast<ccomp::myCast::EnumAttr>(attr1)) {
        if (auto enumAttr2 = llvm::dyn_cast<ccomp::myCast::EnumAttr>(attr2)) {
            std::string enumName = enumAttr1.getName().getValue().data();
            return enumName.compare(enumAttr2.getName().getValue().data()) == 0;
        }
        return false;
    }
    if (auto unionAttr1 = llvm::dyn_cast<ccomp::myCast::UnionAttr>(attr1)) {
        if (auto unionAttr2 = llvm::dyn_cast<ccomp::myCast::UnionAttr>(attr2)) {
            std::string unionName = unionAttr1.getName().getValue().data();
            return unionName.compare(unionAttr2.getName().getValue().data()) == 0;
        }
        return false;
    }
    if (auto structAttr1 = llvm::dyn_cast<ccomp::myCast::StructAttr>(attr1)) {
        if (auto structAttr2 = llvm::dyn_cast<ccomp::myCast::StructAttr>(attr2)) {
            std::string structName = structAttr1.getName().getValue().data();
            return structName.compare(structAttr2.getName().getValue().data()) == 0;
        }
        return false;

    }
    if (auto voidAttr1 = llvm::dyn_cast<ccomp::myCast::VoidAttr>(attr1)) {
        if (auto voidAttr2 = llvm::dyn_cast<ccomp::myCast::VoidAttr>(attr2))
            return true;
        return false;
    }
    return false;
}

#endif // MYCAST_ATTR_H__