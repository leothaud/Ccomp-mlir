//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef CAST_OPS
#define CAST_OPS

include "CastDialect.td"
include "Dialect/Cast/CastOpsTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"



def Cast_ProgramOp : Cast_Op<"program"> {
    let arguments = (ins Variadic<Type<Cast_ProgramItemPred>>:$items);
    let results = (outs Cast_ProgramNodeType:$res);
    let assemblyFormat = [{
        $items `:` type($items) attr-dict
    }];
}

def Cast_FunDefOp : Cast_Op<"funDef"> {
    let arguments = (ins Type<Cast_FunProtoPred>:$proto,
                         Type<Cast_CompoundStatementPred>:$body);
    let results = (outs Cast_FunDefNodeType:$res);
    let assemblyFormat = [{
        $proto `:` type($proto) $body `:` type($body) attr-dict
    }];
}

def Cast_FunProtoOp : Cast_Op<"funProto"> {
    let arguments = (ins I1:$isInlined,
                         Type<Cast_TypePred>:$returnType,
                         Builtin_IntegerAttr:$stars,
                         Variadic<Type<Cast_ArgumentPred>>:$args
                    );
    let results = (outs Cast_FunProtoNodeType:$res);
    let assemblyFormat = [{
        $isInlined $returnType `:` type($returnType) $stars $args `:` type($args) attr-dict
    }];
}

def Cast_ArgumentOp : Cast_Op<"argument"> {
    let arguments = (ins Type<Cast_TypePred>:$varType,
                         Builtin_IntegerAttr:$stars,
                         Builtin_StringAttr:$name,
                         Variadic<Type<Cast_ExpressionPred>>:$sizes);
    let results = (outs Cast_ArgumentNodeType:$res);
    let assemblyFormat = [{
        $varType `:` type($varType) $stars $name $sizes `:` type($sizes) attr-dict
    }];
}

def Cast_VarDeclOp : Cast_Op<"varDecl"> {
    let arguments = (ins Type<Cast_TypePred>:$varType,
                         Variadic<Type<Cast_BaseVarDeclPred>>:$decls);
    let results = (outs Cast_VarDeclNodeType:$res);
    let assemblyFormat = [{
        $varType `:` type($varType) $decls `:` type($decls) attr-dict
    }];
}

def Cast_BaseVarDeclOp : Cast_Op<"baseVarDecl"> {
    let arguments = (ins Builtin_IntegerAttr:$stars,
                         Builtin_StringAttr:$name,
                         Variadic<Type<Cast_ExpressionPred>>:$sizes);
    let results = (outs Cast_BaseVarDeclNodeType);
    let assemblyFormat = [{
        $stars $name $sizes `:` type($sizes) attr-dict
    }];
}

def Cast_UnionDefOp : Cast_Op<"unionDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_VarDeclPred>>:$decls);
    let results = (outs Cast_UnionDefNodeType:$res);
    let assemblyFormat = [{
        $name $decls `:` type($decls) attr-dict
    }];
}

def Cast_StructDefOp : Cast_Op<"structDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_VarDeclPred>>:$decls);
    let results = (outs Cast_StructDefNodeType:$res);
    let assemblyFormat = [{
        $name $decls `:` type($decls) attr-dict
    }];
}

def Cast_EnumDefOp : Cast_Op<"enumDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_EnumItemPred>>:$items);
    let results = (outs Cast_EnumDefNodeType:$res);
    let assemblyFormat = [{
        $name $items `:` type($items) attr-dict
    }];
}

def Cast_AliasDefOp : Cast_Op<"aliasDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Type<Cast_TypePred>:$baseType);
    let results = (outs Cast_AliasDefNodeType:$results);
    let assemblyFormat = [{
        $name $baseType `:` type($baseType) attr-dict
    }];
}

def Cast_EnumItemOp : Cast_Op<"enumItem"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Optional<Type<Cast_ConditionalExpressionPred>>:$value);
    let results = (outs Cast_EnumItemNodeType:$res);
    let assemblyFormat = [{
        $name $value `:` type($value) attr-dict
    }];
}

def Cast_LabeledStatementOp : Cast_Op<"labeledStatement"> {
    let arguments = (ins Builtin_StringAttr:$label,
                         Type<Cast_StatementPred>:$stmt);
    let results = (outs Cast_LabeledStatementNodeType:$res);
    let assemblyFormat = [{
        $label $stmt `:` type($stmt) attr-dict
    }];
}

def Cast_CompoundStatmentOp : Cast_Op<"compoundStatement"> {
    let arguments = (ins Variadic<Type<Cast_StatementPred>>:$stmt);
    let results = (outs Cast_CompoundStatementNodeType:$res);
    let assemblyFormat = [{
        $stmt `:` type($stmt) attr-dict
    }];
}

def Cast_VarDeclStatementOp : Cast_Op<"varDecl"> {
    let arguments = (ins Type<Cast_VarDeclPred>:$decl);
    let results = (outs Cast_VarDeclStatementNodeType:$res);
    let assemblyFormat = [{
        $decl `:` type($decl) attr-dict
    }];
}

def Cast_ExpressionStatementOp : Cast_Op<"expressionStatement"> {
    let arguments = (ins Optional<Type<Cast_ExpressionPred>>:$value);
    let results = (outs Cast_ExpressionStatementNodeType:$res);
    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];
}

def Cast_IfStatementOp : Cast_Op<"ifStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$thenPart,
                         Optional<Type<Cast_StatementPred>>:$elsePart);
    let results = (outs Cast_IfStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $thenPart `:` type($thenPart) $elsePart `:` type($elsePart) attr-dict
    }];
}

def Cast_SwitchStatementOp : Cast_Op<"switchStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Variadic<Type<Cast_SwitchItemPred>>:$items);
    let results = (outs Cast_SwitchStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $items `:` type($items) attr-dict
    }];
}

def Cast_SwitchCaseItemOp : Cast_Op<"SwitchCaseItem"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_SwitchCaseItemNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $body `:` type($body) attr-dict
    }];
}

def Cast_SwitchDefaultItemOp : Cast_Op<"switchDefaultItem"> {
    let arguments = (ins Type<Cast_StatementPred>:$body);
    let results = (outs Cast_SwitchDefaultItemNodeType:$res);
    let assemblyFormat = [{
        $body `:` type($body) attr-dict
    }];
}

def Cast_WhileStatementOp : Cast_Op<"whileStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_WhileStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $body `:` type($body) attr-dict
    }];
}

def Cast_DoWhileStatementOp : Cast_Op<"doWhileStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_DoWhileStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $body `:` type($body) attr-dict
    }];
}

def Cast_ForStatementOp : Cast_Op<"forStatement", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Type<Cast_VarDeclPred>>:$init,
                         Optional<Type<Cast_ExpressionPred>>:$cond,
                         Optional<Type<Cast_ExpressionPred>>:$step,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_ForStatementNodeType:$res);
    let assemblyFormat = [{
        $init `:` type($init) $cond `:` type($cond) $step `:` type($step) $body `:` type($body) attr-dict
    }];
}

def Cast_GotoStatementOp : Cast_Op<"gotoStatement"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_GotoStatementNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_ContinueStatementOp : Cast_Op<"continueStatement"> {
    let arguments = (ins );
    let results = (outs Cast_ContinueStatementNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_BreakStatementOp : Cast_Op<"breakStatement"> {
    let arguments = (ins );
    let results = (outs Cast_BreakStatementNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ReturnStatementOp : Cast_Op<"returnStatement"> {
    let arguments = (ins Optional<Type<Cast_ExpressionPred>>:$value);
    let results = (outs Cast_ReturnStatementNodeType:$res);
    let assemblyFormat = [{
        $value `:` type($value) attr-dict
    }];
}

def Cast_TypeOp : Cast_Op<"Type"> {
    let arguments = (ins Variadic<Type<Cast_TypeModifierPred>>:$modifiers,
                         Type<Cast_BaseTypePred>:$baseType);
    let results = (outs Cast_TypeNodeType:$res);
    let assemblyFormat = [{
        $modifiers `:` type($modifiers) $baseType `:` type($baseType) attr-dict
    }];
}

def Cast_ExternTypeModifierOp : Cast_Op<"externTypeModifier"> {
    let arguments = (ins );
    let results = (outs Cast_ExternTypeModifierNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_StaticTypeModifierOp : Cast_Op<"staticTypeModifier"> {
    let arguments = (ins );
    let results = (outs Cast_StaticTypeModifierNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_RegisterTypeModifierOp : Cast_Op<"RegisterTypeModifier"> {
    let arguments = (ins );
    let results = (outs Cast_RegisterTypeModifierNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ConstTypeModifierOp : Cast_Op<"constTypeModifier"> {
    let arguments = (ins );
    let results = (outs Cast_ConstTypeModifierNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_VolatileTypeModifierOp : Cast_Op<"volatileTypeModifier"> {
    let arguments = (ins );
    let results = (outs Cast_VolatileTypeModifierNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}


def Cast_VoidTypeOp : Cast_Op<"voidType"> {
    let arguments = (ins );
    let results = (outs Cast_VoidTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_UnsignedLongLongTypeOp : Cast_Op<"unsignedLongLongType"> {
    let arguments = (ins );
    let results = (outs Cast_UnsignedLongLongTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_UnsignedLongTypeOp : Cast_Op<"unsignedLongType"> {
    let arguments = (ins );
    let results = (outs Cast_UnsignedLongTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_UnsignedShortTypeOp : Cast_Op<"unsignedShortType"> {
    let arguments = (ins );
    let results = (outs Cast_UnsignedShortTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_UnsignedCharTypeOp : Cast_Op<"unsignedCharType"> {
    let arguments = (ins );
    let results = (outs Cast_UnsignedCharTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_UnsignedIntTypeOp : Cast_Op<"unsignedIntType"> {
    let arguments = (ins );
    let results = (outs Cast_UnsignedIntTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_CharTypeOp : Cast_Op<"charType"> {
    let arguments = (ins );
    let results = (outs Cast_CharTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ShortTypeOp : Cast_Op<"shortType"> {
    let arguments = (ins );
    let results = (outs Cast_ShortTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_IntTypeOp : Cast_Op<"intType"> {
    let arguments = (ins );
    let results = (outs Cast_IntTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LongLongTypeOp : Cast_Op<"longLongType"> {
    let arguments = (ins );
    let results = (outs Cast_LongLongTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LongTypeOp : Cast_Op<"longType"> {
    let arguments = (ins );
    let results = (outs Cast_LongTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_FloatTypeOp : Cast_Op<"floatType"> {
    let arguments = (ins );
    let results = (outs Cast_FloatTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_DoubleTypeOp : Cast_Op<"doubleType"> {
    let arguments = (ins );
    let results = (outs Cast_DoubleTypeNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_EnumTypeOp : Cast_Op<"enumType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_EnumTypeNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_StructTypeOp : Cast_Op<"structType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_StructTypeNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_UnionTypeOp : Cast_Op<"unionType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_UnionTypeNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_AliasTypeOp : Cast_Op<"aliasType"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_AliasTypeNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_ExpressionOp : Cast_Op<"Expression"> {
    let arguments = (ins Variadic<Type<Cast_AssignmentExpressionPred>>:$expr);
    let results = (outs Cast_ExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_AssignmentExpressionOp : Cast_Op<"AssignmentExpression"> {
    let arguments = (ins Type<Cast_UnaryExpressionPred>:$lvalue,
                         Type<Cast_AssignOperatorPred>:$op,
                         Type<Cast_AssignmentExpressionPred>:$rvalue);
    let results = (outs Cast_AssignOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lvalue `:` type($lvalue) $op `:` type($op) $rvalue `:` type($rvalue) attr-dict
    }];
}


def Cast_EqOpOp : Cast_Op<"eqOp"> {
    let arguments = (ins );
    let results = (outs Cast_EqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_StarEqOpOp : Cast_Op<"starEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_StarEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_DivEqOpOp : Cast_Op<"divEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_DivEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ModuloEqOpOp : Cast_Op<"moduloEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_ModuloEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_PlusEqOpOp : Cast_Op<"plusEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_PlusEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_MinusEqOpOp : Cast_Op<"minusEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_MinusEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LeftShiftEqOpOp : Cast_Op<"leftShiftEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_LeftShiftEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_RightShiftEqOpOp : Cast_Op<"rightShiftEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_RightShiftEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_AndEqOpOp : Cast_Op<"andEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_AndEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_XorEqOpOp : Cast_Op<"xorEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_XorEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_OrEqOpOp : Cast_Op<"orEqOp"> {
    let arguments = (ins );
    let results = (outs Cast_OrEqOpNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ConditionalExpressionOp : Cast_Op<"conditionalExpression"> {
    let arguments = (ins Type<Cast_LogicalOrExpressionPred>:$cond,
                         Type<Cast_ExpressionPred>:$thenPart,
                         Type<Cast_ExpressionPred>:$elsePart);
    let results = (outs Cast_ConditionalOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) $thenPart `:` type($thenPart) $elsePart `:` type($elsePart) attr-dict
    }];
}

def Cast_LogicalOrExpressionOp : Cast_Op<"logicalOrExpression"> {
    let arguments = (ins Variadic<Type<Cast_LogicalAndExpressionPred>>:$expr);
    let results = (outs Cast_LogicalOrExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_LogicalAndExpressionOp : Cast_Op<"logicalAndExpression"> {
    let arguments = (ins Variadic<Type<Cast_OrExpressionPred>>:$expr);
    let results = (outs Cast_LogicalAndExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_OrExpressionOp : Cast_Op<"orExpression"> {
    let arguments = (ins Variadic<Type<Cast_XorExpressionPred>>:$expr);
    let results = (outs Cast_OrExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_XorExpressionOp : Cast_Op<"xorExpression"> {
    let arguments = (ins Variadic<Type<Cast_AndExpressionPred>>:$expr);
    let results = (outs Cast_XorExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_AndExpressionOp : Cast_Op<"andExpression"> {
    let arguments = (ins Variadic<Type<Cast_EqualityExpressionPred>>:$expr);
    let results = (outs Cast_AndExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_EqualityExpressionOp : Cast_Op<"equalityExpression"> {
    let arguments = (ins Type<Cast_EqualityExpressionPred>:$lval,
                         Type<Cast_EqualityOperatorPred>:$op,
                         Type<Cast_RelationalExpressionPred>:$rval);
    let results = (outs Cast_EqualOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lval `:` type($lval) $op `:` type($op) $rval `:` type($rval) attr-dict
    }];
}

def Cast_EqualOperatorOp : Cast_Op<"equalOperator"> {
    let arguments = (ins );
    let results = (outs Cast_EqualOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_NotEqualOperatorOp : Cast_Op<"notEqualOperator"> {
    let arguments = (ins );
    let results = (outs Cast_NotEqualOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_RelationalExpressionOp : Cast_Op<"relationalExpression"> {
    let arguments = (ins Type<Cast_RelationalExpressionPred>:$lval,
                         Type<Cast_RelationalOperatorPred>:$op,
                         Type<Cast_ShiftExpressionPred>:$rval);
    let results = (outs Cast_RelOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lval `:` type($lval) $op `:` type($op) $rval `:` type($rval) attr-dict
    }];
}

def Cast_GeOperatorOp : Cast_Op<"geOperator"> {
    let arguments = (ins );
    let results = (outs Cast_GeOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LeOperatorOp : Cast_Op<"leOperator"> {
    let arguments = (ins );
    let results = (outs Cast_LeOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_GtOperatorOp : Cast_Op<"gtOperator"> {
    let arguments = (ins );
    let results = (outs Cast_GtOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LtOperatorOp : Cast_Op<"ltOperator"> {
    let arguments = (ins );
    let results = (outs Cast_LtOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ShiftExpressionOp : Cast_Op<"shiftExpression"> {
    let arguments = (ins Type<Cast_ShiftExpressionPred>:$lval,
                         Type<Cast_ShiftOperatorPred>:$op,
                         Type<Cast_AdditiveExpressionPred>:$rval);
    let results = (outs Cast_ShiftOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lval `:` type($lval) $op `:` type($op) $rval `:` type($rval) attr-dict
    }];
}

def Cast_LshiftOperatorOp : Cast_Op<"lshiftOperator"> {
    let arguments = (ins );
    let results = (outs Cast_LshiftOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_RshiftOperatorOp : Cast_Op<"rshiftOperator"> {
    let arguments = (ins );
    let results = (outs Cast_RshiftOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_AdditiveExpressionOp : Cast_Op<"AdditiveExpression"> {
    let arguments = (ins Type<Cast_AdditiveExpressionPred>:$lval,
                         Type<Cast_AdditiveOperatorPred>:$op,
                         Type<Cast_MultiplicativeExpressionPred>:$rval);
    let results = (outs Cast_AddOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lval `:` type($lval) $op `:` type($op) $rval `:` type($rval) attr-dict
    }];
}

def Cast_PlusOperatorOp : Cast_Op<"plusOperator"> {
    let arguments = (ins );
    let results = (outs Cast_PlusOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_MinusOperatorOp : Cast_Op<"minusOperator"> {
    let arguments = (ins );
    let results = (outs Cast_MinusOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_MultiplicativeExpressionOp : Cast_Op<"multiplicativeExpression"> {
    let arguments = (ins Type<Cast_MultiplicativeExpressionPred>:$lval,
                         Type<Cast_MultiplicativeOperatorPred>:$op,
                         Type<Cast_CastExpressionPred>:$rval);
    let results = (outs Cast_MulOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lval `:` type($lval) $op `:` type($op) $rval `:` type($rval) attr-dict
    }];
}

def Cast_DivOperatorOp : Cast_Op<"divOperator"> {
    let arguments = (ins );
    let results = (outs Cast_DivOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_ModuloOperatorOp : Cast_Op<"moduloOperator"> {
    let arguments = (ins );
    let results = (outs Cast_ModuloOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_CastExpressionOp : Cast_Op<"castExpression"> {
    let arguments = (ins Variadic<Type<Cast_TypePred>>:$types,
                         Type<Cast_UnaryExpressionPred>:$expr);
    let results = (outs Cast_CastExpressionNodeType:$res);
    let assemblyFormat = [{
        $types `:` type($types) $expr `:` type($expr) attr-dict
    }];
}

def Cast_UnopExpresionOp : Cast_Op<"unopExpresion"> {
    let arguments = (ins Type<Cast_UnaryOperatorPred>:$op,
                         Type<Cast_UnaryExpressionPred>:$expr);
    let results = (outs Cast_UnopExpressionNodeType:$res);
    let assemblyFormat = [{
        $op `:` type($op) $expr `:` type($expr) attr-dict
    }];
}


def Cast_IncrOperatorOp : Cast_Op<"incrOperator"> {
    let arguments = (ins );
    let results = (outs Cast_IncrOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_DecrOperatorOp : Cast_Op<"decrOperator"> {
    let arguments = (ins );
    let results = (outs Cast_DecrOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_AddrofOperatorOp : Cast_Op<"addrofOperator"> {
    let arguments = (ins );
    let results = (outs Cast_AddrofOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_DerefOperatorOp : Cast_Op<"derefOperator"> {
    let arguments = (ins );
    let results = (outs Cast_DerefOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_PositiveOperatorOp : Cast_Op<"positiveOperator"> {
    let arguments = (ins );
    let results = (outs Cast_PositiveOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_NegativeOperatorOp : Cast_Op<"negativeOperator"> {
    let arguments = (ins );
    let results = (outs Cast_NegativeOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_NotOperatorOp : Cast_Op<"notOperator"> {
    let arguments = (ins );
    let results = (outs Cast_NotOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_LnotOperatorOp : Cast_Op<"lnotOperator"> {
    let arguments = (ins );
    let results = (outs Cast_LnotOperatorNodeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def Cast_SizeofExpressionsOp : Cast_Op<"sizeofExpressions", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Type<Cast_UnaryExpressionPred>>:$expr,
                         Optional<Type<Cast_TypePred>>:$typeExpr);
    let results = (outs Cast_SizeofExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $typeExpr `:` type($typeExpr) attr-dict
    }];
}

def Cast_PrimaryPostfixExpressionOp : Cast_Op<"primaryPostfixExpression"> {
    let arguments = (ins Type<Cast_PrimaryExpressionPred>:$expr);
    let results = (outs Cast_PrimaryPostfixExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_ArrayExpressionOp : Cast_Op<"arrayExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr,
                         Type<Cast_ExpressionPred>:$index);
    let results = (outs Cast_ArrayExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $index `:` type($index) attr-dict
    }];
}

def Cast_FunCallExpressionOp : Cast_Op<"funCallExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr,
                         Variadic<Type<Cast_AssignmentExpressionPred>>:$args);
    let results = (outs Cast_FunCallExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $args `:` type($args) attr-dict
    }];
}

def Cast_FieldExpressionOp : Cast_Op<"fieldExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr,
                         Builtin_StringAttr:$field);
    let results = (outs Cast_FieldExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $field attr-dict
    }];
}

def Cast_PtrFieldExpressionOp : Cast_Op<"ptrFieldExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr,
                         Builtin_StringAttr:$field);
    let results = (outs Cast_PtrFieldExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $field attr-dict
    }];
}

def Cast_PostincrExpressionOp : Cast_Op<"postincrExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr);
    let results = (outs Cast_PostincrExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_PostdecrExpressionOp : Cast_Op<"postdecrExpression"> {
    let arguments = (ins Type<Cast_PostfixExpressionPred>:$expr);
    let results = (outs Cast_PostdecrExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_VarExpressionOp : Cast_Op<"varExpression"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs Cast_VarExpressionNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict
    }];
}

def Cast_IntExpressionOp : Cast_Op<"intExpression"> {
    let arguments = (ins Builtin_IntegerAttr:$val);
    let results = (outs Cast_IntExpressionNodeType:$res);
    let assemblyFormat = [{
        $val attr-dict
    }];
}

def Cast_FloatExpressionOp : Cast_Op<"floatExpression"> {
    let arguments = (ins F64:$val);
    let results = (outs Cast_FloatExpressionNodeType:$res);
    let assemblyFormat = [{
        $val attr-dict
    }];
}

def Cast_ParentExpressionOp : Cast_Op<"parentExpression"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$expr);
    let results = (outs Cast_ParentExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) attr-dict
    }];
}

def Cast_StructureExpressionOp : Cast_Op<"structureExpression"> {
    let arguments = (ins Variadic<Type<Cast_AssignmentExpressionPred>>:$val);
    let results = (outs Cast_StructureExpressionNodeType:$res);
    let assemblyFormat = [{
        $val `:` type($val) attr-dict
    }];
}

def Cast_GenericExpressionOp : Cast_Op<"genericExpression"> {
    let arguments = (ins Type<Cast_AssignmentExpressionPred>:$expr,
                         Variadic<Type<Cast_GenericItemPred>>:$cases);
    let results = (outs Cast_GenericExpressionNodeType:$res);
    let assemblyFormat = [{
        $expr `:` type($expr) $cases `:` type($cases) attr-dict
    }];
}

def Cast_TypeGenericItemOp : Cast_Op<"typeGenericItem"> {
    let arguments = (ins Type<Cast_TypePred>:$typeExpr,
                         Type<Cast_AssignmentExpressionPred>:$body);
    let results = (outs Cast_TypeGenericItemNodeType:$res);
    let assemblyFormat = [{
        $typeExpr `:` type($typeExpr) $body `:` type($body) attr-dict
    }];
}

def Cast_DefaultGenericItemOp : Cast_Op<"defaultGenericItem"> {
    let arguments = (ins Type<Cast_AssignmentExpressionPred>:$body);
    let results = (outs Cast_DefaultGenericItemNodeType:$res);
    let assemblyFormat = [{
        $body `:` type($body) attr-dict
    }];
}

#endif // CAST_OPS