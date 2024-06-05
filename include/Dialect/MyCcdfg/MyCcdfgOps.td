//===----------------------------MyCcdfgOps.td-----------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_OPS
#define MYCCDFG_OPS

include "Dialect/MyCcdfg/MyCcdfgDialect.td"
include "Dialect/MyCcdfg/MyCcdfgOpsTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"

def MyCcdfg_ProgramOp : MyCcdfg_Op<"program", [AttrSizedOperandSegments]> {
    let arguments = (ins Variadic<Type<MyCcdfg_TypeDefPred>>:$typeDefs,
                         Type<MyCcdfg_ScopePred>:$externScope,
                         Type<MyCcdfg_ScopePred>:$globalScope,
                         Variadic<Type<MyCcdfg_AffectInstrPred>>:$globalInit,
                         Variadic<Type<MyCcdfg_FunDeclPred>>:$funDecls,
                         Variadic<Type<MyCcdfg_FunDefPred>>:$funDefs
                         );
    let results = (outs MyCcdfg_ProgramType:$res);
    let assemblyFormat = [{
        `(` 
        (`(` $typeDefs^ `:` type($typeDefs) `)`)?
        `,` `(` $externScope `:` type($externScope) `)`
        `,` `(` $globalScope `:` type($globalScope) `)`
        `,` (`(` $globalInit^ `:` type($globalInit) `)`)?
        `,` (`(` $funDecls^ `:` type($funDecls) `)`)?
        `,` (`(` $funDefs^ `:` type($funDefs) `)`)?
         `)` attr-dict
    }];
}

def MyCcdfg_ScopeOp : MyCcdfg_Op<"scope"> {
    let arguments = (ins Variadic<Type<MyCcdfg_TypePred>>:$types,
                         StringArrayAttr:$names);
    let results = (outs MyCcdfg_ScopeType:$res);
    let assemblyFormat = [{
        `(`
        (`(` $types^ `:` type($types) `)`)?
        `,` $names
        `)` attr-dict
    }];
}

def MyCcdfg_FunDeclOp : MyCcdfg_Op<"funDecl"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$returnType,
                         Builtin_StringAttr:$funName,
                         Variadic<Type<MyCcdfg_TypePred>>:$argTypes);
    let results = (outs MyCcdfg_FunDeclType:$res);
    let assemblyFormat = [{
        `(`
        `(` $returnType `:` type($returnType) `)`
        `,` $funName
        `,` (`(` $argTypes^ `:` type($argTypes) `)`)?
        `)` attr-dict
    }];
}

def MyCcdfg_FunDefOp : MyCcdfg_Op<"funDef"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$returnType,
                         Builtin_StringAttr:$funName,
                         Type<MyCcdfg_ScopePred>:$parametersScope,
                         Type<MyCcdfg_ScopePred>:$localsScope,
                         Variadic<Type<MyCcdfg_BasicBlockPred>>:$blocks,
                         Type<MyCcdfg_InputTransitionPred>:$inputTransition);
    let results = (outs MyCcdfg_FunDefType:$res);
    let assemblyFormat = [{
        `(`
        `(` $returnType `:` type($returnType) `)`
        `,` $funName
        `,` `(` $parametersScope `:` type($parametersScope) `)`
        `,` `(` $localsScope `:` type($localsScope) `)`
        `,` (`(` $blocks^ `:` type($blocks) `)`)?
        `,` `(` $inputTransition `:` type($inputTransition) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_UnionDefOp : MyCcdfg_Op<"unionDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCcdfg_TypePred>>:$fieldTypes,
                         StringArrayAttr:$fieldNames);
    let results = (outs MyCcdfg_UnionDefType:$res);
    let assemblyFormat = [{
        `(`
        $name `,` (`(` $fieldTypes^ `:` type($fieldTypes) `)`)? `,` $fieldNames
        `)` attr-dict
    }];
}

def MyCcdfg_StructDefOp : MyCcdfg_Op<"structDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCcdfg_TypePred>>:$fieldTypes,
                         StringArrayAttr:$fieldNames);
    let results = (outs MyCcdfg_StructDefType:$res);
    let assemblyFormat = [{
        `(`
        $name `,` (`(` $fieldTypes^ `:` type($fieldTypes) `)`)? `,` $fieldNames
        `)` attr-dict
    }];
}

def MyCcdfg_EnumDefOp : MyCcdfg_Op<"enumDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         StringArrayAttr:$fieldNames,
                         I64ArrayAttr:$fieldValues);
    let results = (outs MyCcdfg_EnumDefType:$res);
    let assemblyFormat = [{
        `(` $name `,` $fieldNames `,` $fieldValues `)` attr-dict
    }];
}

def MyCcdfg_ModifiedTypeOp : MyCcdfg_Op<"modifiedType"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$baseType,
                         Builtin_IntegerAttr:$isStatic,
                         Builtin_IntegerAttr:$isConst,
                         Builtin_IntegerAttr:$isVolatile);
    let results = (outs MyCcdfg_ModifiedTypeType:$res);
    let assemblyFormat = [{
        `(`
        `(` $baseType `:` type($baseType) `)`
        `,` $isStatic
        `,` $isConst
        `,` $isVolatile
        `)` attr-dict
    }];
}

def MyCcdfg_PointerTypeOp : MyCcdfg_Op<"pointerType"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$baseType);
    let results = (outs MyCcdfg_PointerTypeType:$res);
    let assemblyFormat = [{
        `(` $baseType `:` type($baseType) `)` attr-dict
    }];
}

def MyCcdfg_ArrayTypeOp : MyCcdfg_Op<"arrayType"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$baseType,
                         Type<MyCcdfg_ExpressionPred>:$size);
    let results = (outs MyCcdfg_ArrayTypeType:$res);
    let assemblyFormat = [{
        `(`
        `(` $baseType `:` type($baseType) `)`
        `,` `(` $size `:` type($size) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_VoidTypeOp : MyCcdfg_Op<"voidType"> {
    let arguments = (ins );
    let results = (outs MyCcdfg_VoidTypeType:$res);
    let assemblyFormat = [{ attr-dict }];
}

def MyCcdfg_IntTypeOp : MyCcdfg_Op<"intType"> {
    let arguments = (ins );
    let results = (outs MyCcdfg_IntTypeType:$res);
    let assemblyFormat = [{ attr-dict }];
}

def MyCcdfg_FloatTypeOp : MyCcdfg_Op<"floatType"> {
    let arguments = (ins );
    let results = (outs MyCcdfg_FloatTypeType:$res);
    let assemblyFormat = [{ attr-dict }];
}

def MyCcdfg_DoubleTypeOp : MyCcdfg_Op<"doubleType"> {
    let arguments = (ins );
    let results = (outs MyCcdfg_DoubleTypeType:$res);
    let assemblyFormat = [{ attr-dict }];
}

def MyCcdfg_EnumTypeOp : MyCcdfg_Op<"enumType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs MyCcdfg_EnumTypeType:$res);
    let assemblyFormat = [{
        `(` $name `)` attr-dict
    }];
}

def MyCcdfg_StructTypeOp : MyCcdfg_Op<"structType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs MyCcdfg_StructTypeType:$res);
    let assemblyFormat = [{
        `(` $name `)` attr-dict
    }];
}

def MyCcdfg_UnionTypeOp : MyCcdfg_Op<"unionType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs MyCcdfg_UnionTypeType:$res);
    let assemblyFormat = [{
        `(` $name `)` attr-dict
    }];
}

def MyCcdfg_MuxExpressionOp : MyCcdfg_Op<"muxExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$cond,
                         Type<MyCcdfg_ExpressionPred>:$thenPart,
                         Type<MyCcdfg_ExpressionPred>:$elsePart);
    let results = (outs MyCcdfg_MuxExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $cond `:` type($cond) `)`
        `,` `(` $thenPart `:` type($thenPart) `)`
        `,` `(` $elsePart `:` type($elsePart) `)`
        `)` attr-dict
    }];

}

def MyCcdfg_LandExpressionOp : MyCcdfg_Op<"landExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_LandExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_LorExpressionOp : MyCcdfg_Op<"lorExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_LorExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_AndExpressionOp : MyCcdfg_Op<"andExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_AndExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_OrExpressionOp : MyCcdfg_Op<"orExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_OrExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_XorExpressionOp : MyCcdfg_Op<"xorExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_XorExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_EqExpressionOp : MyCcdfg_Op<"eqExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_EqExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_NeqExpressionOp : MyCcdfg_Op<"neqExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_NeqExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_GeExpressionOp : MyCcdfg_Op<"geExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_GeExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_GtExpressionOp : MyCcdfg_Op<"gtExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_GtExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_LeExpressionOp : MyCcdfg_Op<"leExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_LeExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_LtExpressionOp : MyCcdfg_Op<"ltExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_LtExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_LshiftExpressionOp : MyCcdfg_Op<"lshiftExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_LshiftExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_RshiftExpressionOp : MyCcdfg_Op<"rshiftExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_RshiftExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_AddExpressionOp : MyCcdfg_Op<"addExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_AddExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_SubExpressionOp : MyCcdfg_Op<"SubExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_SubExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_CastExpressionOp : MyCcdfg_Op<"castExpression"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$newType,
                         Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_CastExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $newType `:` type($newType) `)`
        `,` `(` $expr `:` type($expr) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_MultExpressionOp : MyCcdfg_Op<"multExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_MultExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_DivExpressionOp : MyCcdfg_Op<"divExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_DivExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_ModuloExpressionOp : MyCcdfg_Op<"moduloExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$lop,
                         Type<MyCcdfg_ExpressionPred>:$rop);
    let results = (outs MyCcdfg_ModuloExpressionType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lop `:` type($lop) `)`
        `,` `(` $rop `:` type($rop) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_SizeofExprExpressionOp : MyCcdfg_Op<"sizeofExprExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_SizeofExprExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_SizeofTypeExpressionOp : MyCcdfg_Op<"sizeofTypeExpression"> {
    let arguments = (ins Type<MyCcdfg_TypePred>:$expr);
    let results = (outs MyCcdfg_SizeofTypeExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_DerefExpressionOp : MyCcdfg_Op<"derefExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_DerefExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_AddrofExpressionOp : MyCcdfg_Op<"addrofExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_AddrofExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_NegExpressionOp : MyCcdfg_Op<"negExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_NegExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_NotExpressionOp : MyCcdfg_Op<"notExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_NotExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_LnotExpressionOp : MyCcdfg_Op<"lnotExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr);
    let results = (outs MyCcdfg_LnotExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)` attr-dict
    }];
}

def MyCcdfg_FieldExpressionOp : MyCcdfg_Op<"fieldExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr,
                         Builtin_StringAttr:$fieldName);
    let results = (outs MyCcdfg_FieldExpressionType:$res);
    let assemblyFormat = [{
        `(` `(` $expr `:` type($expr) `)` `,` $fieldName `)` attr-dict
    }];
}

def MyCcdfg_PtrFieldExpressionOp : MyCcdfg_Op<"ptrFieldExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr,
                         Builtin_StringAttr:$fieldName);
    let results = (outs MyCcdfg_PtrFieldExpressionType:$res);
    let assemblyFormat = [{
        `(` `(` $expr `:` type($expr) `)` `,` $fieldName `)` attr-dict
    }];
}

def MyCcdfg_FunCallExpressionOp : MyCcdfg_Op<"funCallExpression"> {
    let arguments = (ins Builtin_StringAttr:$funName,
                         Variadic<Type<MyCcdfg_ExpressionPred>>:$args);
    let results = (outs MyCcdfg_FunCallExpressionType:$res);
    let assemblyFormat = [{
        `(`
        $funName
        `,` (`(` $args^ `:` type($args) `)`)?
        `)` attr-dict
    }];
}

def MyCcdfg_ArrayExpressionOp : MyCcdfg_Op<"arrayExpression"> {
    let arguments = (ins Type<MyCcdfg_ExpressionPred>:$expr,
                         Type<MyCcdfg_ExpressionPred>:$index);
    let results = (outs MyCcdfg_ArrayExpressionType:$res);
    let assemblyFormat = [{
        `(` `(` $expr `:` type($expr) `)` `,` $index `:` type($index) `)` attr-dict
    }];
}

def MyCcdfg_VarExpressionOp : MyCcdfg_Op<"varExpression"> {
    let arguments = (ins Builtin_StringAttr:$varName);
    let results = (outs MyCcdfg_VarExpressionType:$res);
    let assemblyFormat = [{
        `(` $varName `)` attr-dict
    }];
}

def MyCcdfg_IntConstExpressionOp : MyCcdfg_Op<"intConstExpression"> {
    let arguments = (ins Builtin_IntegerAttr:$val);
    let results = (outs MyCcdfg_IntConstExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `)` attr-dict
    }];
}

def MyCcdfg_FloatConstExpressionOp : MyCcdfg_Op<"floatConstExpression"> {
    let arguments = (ins Builtin_FloatAttr:$val);
    let results = (outs MyCcdfg_FloatConstExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `)` attr-dict
    }];
}

def MyCcdfg_StringConstExpressionOp : MyCcdfg_Op<"stringConstExpression"> {
    let arguments = (ins Builtin_StringAttr:$val);
    let results = (outs MyCcdfg_StringConstExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `)` attr-dict
    }];
}

def MyCcdfg_StructureConstExpressionOp : MyCcdfg_Op<"structureConstExpression"> {
    let arguments = (ins Variadic<Type<MyCcdfg_ExpressionPred>>:$vals);
    let results = (outs MyCcdfg_StructureConstExpressionType:$res);
    let assemblyFormat = [{
        `(` ($vals^ `:` type($vals))?`)` attr-dict
    }];
}

def MyCcdfg_LvalueVarOp : MyCcdfg_Op<"lvalueVar"> {
    let arguments = (ins Builtin_StringAttr:$varName);
    let results = (outs MyCcdfg_LvalueVarType:$res);
    let assemblyFormat = [{
        `(` $varName `)` attr-dict
    }];
}

def MyCcdfg_LvalueArrayOp : MyCcdfg_Op<"lvalueArray"> {
    let arguments = (ins Type<MyCcdfg_LvaluePred>:$base,
                         Type<MyCcdfg_ExpressionPred>:$index);
    let results = (outs MyCcdfg_LvalueArrayType:$res);
    let assemblyFormat = [{
        `(`
        `(` $base `:` type($base) `)`
        `,` `(` $index `:` type($index) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_LvalueDerefOp : MyCcdfg_Op<"lvalueDeref"> {
    let arguments = (ins Type<MyCcdfg_LvaluePred>:$base);
    let results = (outs MyCcdfg_LvalueDerefType:$res);
    let assemblyFormat = [{
        `(` $base `:` type($base) `)`
        attr-dict
    }];
}

def MyCcdfg_LvalueFieldOp : MyCcdfg_Op<"lvalueField"> {
    let arguments = (ins Type<MyCcdfg_LvaluePred>:$base,
                         Builtin_StringAttr:$fieldName);
    let results = (outs MyCcdfg_LvalueFieldType:$res);
    let assemblyFormat = [{
        `(` $base `:` type($base) `)`
        `,` $fieldName
        attr-dict
    }];
}

def MyCcdfg_LvaluePtrFieldOp : MyCcdfg_Op<"lvaluePtrField"> {
    let arguments = (ins Type<MyCcdfg_LvaluePred>:$base,
                         Builtin_StringAttr:$fieldName);
    let results = (outs MyCcdfg_LvaluePtrFieldType:$res);
    let assemblyFormat = [{
        `(` $base `:` type($base) `)`
        `,` $fieldName
        attr-dict
    }];
}

def MyCcdfg_BasicBlockOp : MyCcdfg_Op<"basicBlock", [AttrSizedOperandSegments]> {
    let arguments = (ins Variadic<Type<MyCcdfg_InstrPred>>:$instrs,
                         Variadic<Type<MyCcdfg_TransitionPred>>:$outputTransitions);
    let results = (outs MyCcdfg_BasicBlockType:$res);
    let assemblyFormat = [{
        `(`
        (`(` $instrs^ `:` type($instrs) `)`)?
        `,` ( `(` $outputTransitions^ `:` type($outputTransitions) `)` )?
        `)` attr-dict
    }];
}

def MyCcdfg_FunCallInstrOp : MyCcdfg_Op<"funCallInstr"> {
    let arguments = (ins Builtin_StringAttr:$funName,
                         Variadic<Type<MyCcdfg_ExpressionPred>>:$args);
    let results = (outs MyCcdfg_FunCallInstrType:$res);
    let assemblyFormat = [{
        `(`
        $funName `,` `(` $args `:` type($args) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_AffectInstrOp : MyCcdfg_Op<"affectInstr"> {
    let arguments = (ins Type<MyCcdfg_LvaluePred>:$lvalue,
                         Type<MyCcdfg_ExpressionPred>:$rvalue);
    let results = (outs MyCcdfg_AffectInstrType:$res);
    let assemblyFormat = [{
        `(`
        `(` $lvalue `:` type($lvalue) `)`
        `,` `(` $rvalue `:` type($rvalue) `)`
        `)` attr-dict
    }];
}

def MyCcdfg_InputTransitionOp : MyCcdfg_Op<"inputTransition"> {
    let arguments = (ins Type<MyCcdfg_BasicBlockPred>:$block);
    let results = (outs MyCcdfg_InputTransitionType:$res);
    let assemblyFormat = [{
        `(` $block `:` type($block) `)` attr-dict
    }];
}

def MyCcdfg_OutputTransitionOp : MyCcdfg_Op<"outputTransition"> {
    let arguments = (ins );
    let results = (outs MyCcdfg_OutputTransitionType:$res);
    let assemblyFormat = [{ attr-dict }];
}

def MyCcdfg_UnconditionalTransitionOp : MyCcdfg_Op<"unconditionalTransition"> {
    let arguments = (ins Type<MyCcdfg_BasicBlockPred>:$dest);
    let results = (outs MyCcdfg_UnconditionalTransitionType:$res);
    let assemblyFormat = [{
        `(` $dest `:` type($dest) `)` attr-dict
    }];
}

def MyCcdfg_ConditionalTransitionOp : MyCcdfg_Op<"conditionalTransition"> {
    let arguments = (ins Builtin_StringAttr:$condVar,
                         Type<MyCcdfg_BasicBlockPred>:$dest);
    let results = (outs MyCcdfg_ConditionalTransitionType:$res);
    let assemblyFormat = [{
        `(`
        $condVar
        `,` `(` $dest `:` type($dest) `)`
        `)` attr-dict
    }];
}

#endif // MYCCDFG_OPS