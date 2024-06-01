//===---------------------MyCastOps.td-------------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//


#ifndef MYCAST_OPS
#define MYCAST_OPS

include "Dialect/MyCast/MyCastDialect.td"
include "Dialect/MyCast/MyCastOpsTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"



def MyCast_ProgramOp : MyCast_Op<"program", [AttrSizedOperandSegments]> {
    let arguments = (ins Variadic<Type<MyCast_FunDefPred>>:$funDefs,
                         Variadic<Type<MyCast_VarDeclPred>>:$varDecls,
                         Variadic<Type<MyCast_TypeDefPred>>:$typeDefs);
    let results = (outs MyCast_ProgramType:$res);
    let assemblyFormat = [{
        `(` (`(` $funDefs^ `:` type($funDefs) `)`)?
        `,` (`(` $varDecls^ `:` type($varDecls) `)`)?
        `,` ( `(` $typeDefs^ `:` type($typeDefs) `)`)? `)`
        attr-dict
    }];
}

def MyCast_FunDefOp : MyCast_Op<"funDef"> {
    let arguments = (ins Type<MyCast_FunProtoPred>:$proto,
                         Optional<Type<MyCast_CompoundStatementPred>>:$body);
    let results = (outs MyCast_FunDefType:$res);
    let assemblyFormat = [{
        `(` `(` $proto `:` type($proto) `)`
        `,` (`(` $body^ `:` type($body)`)`)? `)` 
        attr-dict
    }];
}

def MyCast_FunProtoOp : MyCast_Op<"funProto"> {
    let arguments = (ins Type<MyCast_TypePred>:$returnType,
                         Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_ArgumentPred>>:$args);
    let results = (outs MyCast_FunProtoType:$res);
    let assemblyFormat = [{
        `(` `(`$returnType `:` type($returnType) `)` `,` $name `,` (`(`$args^ `:` type($args)`)`)? `)`
        attr-dict
    }];
}

def MyCast_ArgumentOp : MyCast_Op<"argument"> {
    let arguments = (ins Type<MyCast_TypePred>:$argType,
                         Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_ExpressionPred>>:$sizes);
    let results = (outs MyCast_ArgumentType:$res);
    let assemblyFormat = [{
        `(` `(`$argType `:` type($argType) `)` `,` $name `,` (`(` $sizes^ `:`type($sizes) `)`)? `)` 
        attr-dict
    }];
}

def MyCast_VarDeclOp : MyCast_Op<"varDecl"> {
    let arguments = (ins Type<MyCast_TypePred>:$argType,
                         Variadic<Type<MyCast_BaseVarDeclPred>>:$decls);
    let results = (outs MyCast_VarDeclType:$res);
    let assemblyFormat = [{
        `(` `(`$argType type($argType) `)` `,` ( `(` $decls^ `:` type($decls) `)`)? `)` 
        attr-dict
    }];
}

def MyCast_BaseVarDeclOp : MyCast_Op<"baseVarDecl", [AttrSizedOperandSegments]> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_ExpressionPred>>:$sizes,
                         Optional<Type<MyCast_AssignmentExpressionPred>>:$value);
    let results = (outs MyCast_BaseVarDeclType:$res);
    let assemblyFormat = [{
        `(` $name `,` (`(` $sizes^ `:` type($sizes)`)` )? `,` (`(` $value^ `:` type($value)`)`)? `)` 
        attr-dict
    }];
}

def MyCast_UnionDefOp : MyCast_Op<"unionDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_VarDeclPred>>:$decls);
    let results = (outs MyCast_UnionDefType:$res);
    let assemblyFormat = [{
        `(` $name `,` (`(`$decls^ `:` type($decls) `)`)? `)` 
        attr-dict
    }];
}

def MyCast_StructDefOp : MyCast_Op<"structDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_VarDeclPred>>:$decls);
    let results = (outs MyCast_StructDefType:$res);
    let assemblyFormat = [{
        `(` $name `,` (`(`$decls^ `:` type($decls) `)`)? `)` 
        attr-dict
    }];
}

def MyCast_EnumDefOp : MyCast_Op<"enumDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<MyCast_EnumItemPred>>:$items);
    let results = (outs MyCast_EnumDefType:$res);
    let assemblyFormat = [{
        `(` $name `,` (`(`$items^ `:` type($items) `)`)? `)`
        attr-dict
    }];
}

def MyCast_LabeledStatementOp : MyCast_Op<"labeledStatement"> {
    let arguments = (ins Builtin_StringAttr:$label,
                         Type<MyCast_StatementPred>:$stmt);
    let results = (outs MyCast_LabeledStatementType:$res);
    let assemblyFormat = [{
        `(` $label `,` `(` $stmt `:` type($stmt) `)` `)` 
        attr-dict
    }];
}

def MyCast_CompoundStatementOp : MyCast_Op<"compoundStatement"> {
    let arguments = (ins Variadic<Type<MyCast_StatementPred>>:$stmt);
    let results = (outs MyCast_CompoundStatementType:$res);
    let assemblyFormat = [{
        `(` ($stmt^ `:` type($stmt))? `)`
        attr-dict
    }];
}

def MyCast_VarDeclStatementOp : MyCast_Op<"varDeclStatement"> {
    let arguments = (ins Type<MyCast_VarDeclPred>:$decl);
    let results = (outs MyCast_VarDeclStatementType:$res);
    let assemblyFormat = [{
        `(` $decl `:` type($decl) `)`
        attr-dict
    }];
}

def MyCast_ExpressionStatementOp : MyCast_Op<"expressionStatement"> {
    let arguments = (ins Optional<Type<MyCast_ExpressionPred>>:$value);
    let results = (outs MyCast_ExpressionStatementType:$res);
    let assemblyFormat = [{
        `(` ($value^ `:` type($value))? `)`
        attr-dict
    }];
}

def MyCast_IfStatementOp : MyCast_Op<"ifStatement"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$cond,
                         Type<MyCast_StatementPred>:$thenPart,
                         Optional<Type<MyCast_StatementPred>>:$elsePart);
    let results = (outs MyCast_IfStatementType:$res);
    let assemblyFormat = [{
        `(` `(` $cond `:` type($cond) `)`
        `,` `(` $thenPart `:`type($thenPart) `)`
        `,` (`(` $elsePart^ `:` type($elsePart) `)`)? `)` 
        attr-dict
    }];
}

def MyCast_SwitchStatementOp : MyCast_Op<"switchStatement"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$cond,
                         Variadic<Type<MyCast_SwitchItemPred>>:$items);
    let results = (outs MyCast_SwitchStatementType:$res);
    let assemblyFormat = [{
        `(` `(` $cond `:` type($cond) `)`
        `,` (`(`$items^ `:` type($items)`)`)? `)` 
        attr-dict
    }];
}

def MyCast_SwitchCaseItemOp : MyCast_Op<"switchCaseItem"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$cond,
                         Type<MyCast_StatementPred>:$body);
    let results = (outs MyCast_SwitchCaseItemType:$res);
    let assemblyFormat = [{
        `(` `(` $cond `:` type($cond) `)` `,` `(` $body `:` type($body) `)` `)` 
        attr-dict
    }];
}

def MyCast_SwitchDefaultItemOp : MyCast_Op<"switchDefaultItem"> {
    let arguments = (ins Type<MyCast_StatementPred>:$body);
    let results = (outs MyCast_SwitchDefaultItemType:$res);
    let assemblyFormat = [{
        `(` $body `:` type($body) `)`
        attr-dict
    }];
}

def MyCast_WhileStatementOp : MyCast_Op<"whileStatement"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$cond,
                         Type<MyCast_StatementPred>:$body);
    let results = (outs MyCast_WhileStatementType:$res);
    let assemblyFormat = [{
        `(` `(` $cond `:` type($cond) `)` `,` `(` $body `:` type($body) `)` `)`
        attr-dict
    }];
}

def MyCast_DoWhileStatementOp : MyCast_Op<"doWhileStatement"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$cond,
                         Type<MyCast_StatementPred>:$body);
    let results = (outs MyCast_DoWhileStatementType:$res);
    let assemblyFormat = [{
        `(` $cond `:` type($cond) `)` `,` `(` $body `:` type($body) `)` `)` 
        attr-dict
    }];
}

def MyCast_ForStatementOp : MyCast_Op<"forStatement", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Type<MyCast_VarDeclPred>>:$init,
                         Optional<Type<MyCast_ExpressionPred>>:$cond,
                         Optional<Type<MyCast_ExpressionPred>>:$step,
                         Type<MyCast_StatementPred>:$body);
    let results = (outs MyCast_ForStatementType:$res);
    let assemblyFormat = [{
        `(` (`(` $init^ `:` type($init)`)`)?
        `,` (`(` $cond^ `:` type($cond)`)`)?
        `,` (`(` $step^ `:` type($step)`)`)?
        `,` `(` $body `:` type($body) `)` `)` 
        attr-dict
    }];
}

def MyCast_GotoStatementOp : MyCast_Op<"gotoStatement"> {
    let arguments = (ins );
    let results = (outs MyCast_GotoStatementType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ContinueStatementOp : MyCast_Op<"continueStatement"> {
    let arguments = (ins );
    let results = (outs MyCast_ContinueStatementType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_BreakStatementOp : MyCast_Op<"breakStatement"> {
    let arguments = (ins );
    let results = (outs MyCast_BreakStatementType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ReturnStatementOp : MyCast_Op<"returnStatement"> {
    let arguments = (ins Optional<Type<MyCast_ExpressionPred>>:$value);
    let results = (outs MyCast_ReturnStatementType:$res);
    let assemblyFormat = [{
        `(` $value `:` type($value) `)`
        attr-dict
    }];
}

def MyCast_TypeOp : MyCast_Op<"type"> {
    let arguments = (ins Variadic<Type<MyCast_TypeModifierPred>>:$modifiers,
                         Type<MyCast_BaseTypePred>:$bt,
                         Builtin_IntegerAttr:$stars);
    let results = (outs MyCast_TypeType:$res);
    let assemblyFormat = [{
        `(` `(` $modifiers `:` type($modifiers) `)`
        `,` `(` $bt `:` type($bt) `)`
        `,` $stars `)` 
        attr-dict
    }];
}

def MyCast_StaticTypeModifierOp : MyCast_Op<"staticTypeModifier"> {
    let arguments = (ins );
    let results = (outs MyCast_StaticTypeModifierType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ConstTypeModifierOp : MyCast_Op<"constTypeModifier"> {
    let arguments = (ins );
    let results = (outs MyCast_ConstTypeModifierType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ExternTypeModifierOp : MyCast_Op<"externTypeModifier"> {
    let arguments = (ins );
    let results = (outs MyCast_ExternTypeModifierType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_VolatileTypeModifierOp : MyCast_Op<"volatileTypeModifier"> {
    let arguments = (ins );
    let results = (outs MyCast_VolatileTypeModifierType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_VoidTypeOp : MyCast_Op<"voidType"> {
    let arguments = (ins );
    let results = (outs MyCast_VoidTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnsignedLongLongTypeOp : MyCast_Op<"unsignedLongLongType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnsignedLongLongTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnsignedLongTypeOp : MyCast_Op<"unsignedLongType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnsignedLongTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnsignedShortTypeOp : MyCast_Op<"UnsignedShortType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnsignedShortTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnsignedCharTypeOp : MyCast_Op<"UnsignedCharType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnsignedCharTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnsignedIntTypeOp : MyCast_Op<"UnsignedIntType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnsignedIntTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}


def MyCast_LongLongTypeOp : MyCast_Op<"longLongType"> {
    let arguments = (ins );
    let results = (outs MyCast_LongLongTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_LongTypeOp : MyCast_Op<"longType"> {
    let arguments = (ins );
    let results = (outs MyCast_LongTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ShortTypeOp : MyCast_Op<"shortType"> {
    let arguments = (ins );
    let results = (outs MyCast_ShortTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_CharTypeOp : MyCast_Op<"charType"> {
    let arguments = (ins );
    let results = (outs MyCast_CharTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_IntTypeOp : MyCast_Op<"intType"> {
    let arguments = (ins );
    let results = (outs MyCast_IntTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_FloatTypeOp : MyCast_Op<"floatType"> {
    let arguments = (ins );
    let results = (outs MyCast_FloatTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_DoubleTypeOp : MyCast_Op<"doubleType"> {
    let arguments = (ins );
    let results = (outs MyCast_DoubleTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_EnumypeOp : MyCast_Op<"enumType"> {
    let arguments = (ins );
    let results = (outs MyCast_EnumTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_StructTypeOp : MyCast_Op<"structType"> {
    let arguments = (ins );
    let results = (outs MyCast_StructTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_UnionTypeOp : MyCast_Op<"unionType"> {
    let arguments = (ins );
    let results = (outs MyCast_UnionTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_AliasTypeOp : MyCast_Op<"aliasType"> {
    let arguments = (ins );
    let results = (outs MyCast_AliasTypeType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_AssignmentExpressionOp : MyCast_Op<"assignmentExpression"> {
    let arguments = (ins Type<MyCast_UnaryExpressionPred>:$lvalue,
                         Type<MyCast_AssignOperatorPred>:$op,
                         Type<MyCast_AssignmentExpressionPred>:$rvalue);
    let results = (outs MyCast_AssignmentExpressionType:$res);
    let assemblyFormat = [{
        `(` `(` $lvalue `:` type($lvalue) `)`
        `,` `(` $op `:` type($op) `)`
        `,` `(` $rvalue `:` type($rvalue) `)` `)` 
        attr-dict
    }];
}

def MyCast_EqOpOp : MyCast_Op<"eqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_EqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_StarEqOpOp : MyCast_Op<"starEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_StarEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_DivEqOpOp : MyCast_Op<"divEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_DivEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ModuloEqOpOp : MyCast_Op<"moduloEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_ModuloEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_PlusEqOpOp : MyCast_Op<"plusEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_PlusEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_MinusEqOpOp : MyCast_Op<"minusEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_MinusEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_LeftShiftEqOpOp : MyCast_Op<"leftShiftEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_LeftShiftEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_RightShiftEqOpOp : MyCast_Op<"rightShiftEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_RightShiftEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_AndEqOpOp : MyCast_Op<"andEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_AndEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_XorEqOpOp : MyCast_Op<"xorEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_XorEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_OrEqOpOp : MyCast_Op<"orEqOp"> {
    let arguments = (ins );
    let results = (outs MyCast_OrEqOpType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ConditionalExpressionOp : MyCast_Op<"conditionalExpression"> {
    let arguments = (ins Type<MyCast_LogicalOrExpressionPred>:$cond,
                         Type<MyCast_ExpressionPred>:$thenPart,
                         Type<MyCast_ConditionalExpressionPred>:$elsePart);
    let results = (outs MyCast_ConditionalExpressionType:$res);
    let assemblyFormat = [{
        `(` `(`$cond `:` type($cond)`)`
        `,` `(`$thenPart `:` type($thenPart)`)`
        `,` `(`$elsePart `:` type($elsePart)`)``)`
        attr-dict
    }];
}

def MyCast_LogicalOrExpressionOp : MyCast_Op<"logicalOrExpression"> {
    let arguments = (ins Variadic<Type<MyCast_LogicalAndExpressionPred>>:$expr);
    let results = (outs MyCast_LogicalOrExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_LogicalAndExpressionOp : MyCast_Op<"logicalAndExpression"> {
    let arguments = (ins Variadic<Type<MyCast_OrExpressionPred>>:$expr);
    let results = (outs MyCast_LogicalAndExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_OrExpressionOp : MyCast_Op<"orExpression"> {
    let arguments = (ins Variadic<Type<MyCast_XorExpressionPred>>:$expr);
    let results = (outs MyCast_OrExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_XorExpressionOp : MyCast_Op<"xorExpression"> {
    let arguments = (ins Variadic<Type<MyCast_AndExpressionPred>>:$expr);
    let results = (outs MyCast_XorExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_AndExpressionOp : MyCast_Op<"andExpression"> {
    let arguments = (ins Variadic<Type<MyCast_EqualityExpressionPred>>:$expr);
    let results = (outs MyCast_AndExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_EqualityExpressionOp : MyCast_Op<"equalityExpression"> {
    let arguments = (ins Type<MyCast_EqualityExpressionPred>:$lval,
                         Type<MyCast_EqualityOperatorPred>:$op,
                         Type<MyCast_RelationalExpressionPred>:$rval);
    let results = (outs MyCast_EqualityExpressionType:$res);
    let assemblyFormat = [{
        `(` $lval `:` type($lval)`)`
        `,` $op `:` type($op)`)`
        `,` $rval  `:` type($rval)`)``)`
        attr-dict
    }];
}

def MyCast_EqualOperatorOp : MyCast_Op<"equalOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_EqualOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_NotEqualOperatorOp : MyCast_Op<"notEqualOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_NotEqualOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_RelationalExpressionOp : MyCast_Op<"relationalExpression"> {
    let arguments = (ins Type<MyCast_RelationalExpressionPred>:$lval,
                         Type<MyCast_RelationalOperatorPred>:$op,
                         Type<MyCast_ShiftExpressionPred>:$rval);
    let results = (outs MyCast_RelationalExpressionType:$res);
    let assemblyFormat = [{
        `(` $lval `:` type($lval)`)`
        `,` $op `:` type($op)`)`
        `,` $rval  `:` type($rval)`)``)`
        attr-dict
    }];
}

def MyCast_GeOperatorOp : MyCast_Op<"geOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_GeOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_GtOperatorOp : MyCast_Op<"gtOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_GtOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_LeOperatorOp : MyCast_Op<"leOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_LeOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_LtOperatorOp : MyCast_Op<"ltOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_LtOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ShiftExpressionOp : MyCast_Op<"shiftExpression"> {
    let arguments = (ins Type<MyCast_ShiftExpressionPred>:$lval,
                         Type<MyCast_ShiftOperatorPred>:$op,
                         Type<MyCast_AdditiveExpressionPred>:$rval);
    let results = (outs MyCast_ShiftExpressionType:$res);
    let assemblyFormat = [{
        `(` $lval `:` type($lval)`)`
        `,` $op `:` type($op)`)`
        `,` $rval  `:` type($rval)`)``)`
        attr-dict
    }];
}

def MyCast_LshiftOperatorOp : MyCast_Op<"lshiftOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_LshiftOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_RshiftOperatorOp : MyCast_Op<"rshiftOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_RshiftOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_AdditiveExpressionOp : MyCast_Op<"additiveExpression"> {
    let arguments = (ins Type<MyCast_AdditiveExpressionPred>:$lval,
                         Type<MyCast_AdditiveOperatorPred>:$op,
                         Type<MyCast_MultiplicativeExpressionPred>:$rval);
    let results = (outs MyCast_AdditiveExpressionType:$res);
    let assemblyFormat = [{
        `(` $lval `:` type($lval)`)`
        `,` $op `:` type($op)`)`
        `,` $rval  `:` type($rval)`)``)`
        attr-dict
    }];
}

def MyCast_PlusOperatorOp : MyCast_Op<"plusOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_PlusOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_MinusOperatorOp : MyCast_Op<"minusOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_MinusOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_MultiplicativeExpressionOp : MyCast_Op<"MultiplicativeExpression"> {
    let arguments = (ins Type<MyCast_MultiplicativeExpressionPred>:$lval,
                         Type<MyCast_MultiplicativeOperatorPred>:$op,
                         Type<MyCast_CastExpressionPred>:$rval);
    let results = (outs MyCast_MultiplicativeExpressionType:$res);
    let assemblyFormat = [{
        `(` $lval `:` type($lval)`)`
        `,` $op `:` type($op)`)`
        `,` $rval  `:` type($rval)`)``)`
        attr-dict
    }];
}

def MyCast_MultOperatorOp : MyCast_Op<"multOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_MultOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_DivOperatorOp : MyCast_Op<"dvOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_DivOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_ModuloOperatorOp : MyCast_Op<"moduloOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_ModuloOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_CastExpressionOp : MyCast_Op<"castExpression"> {
    let arguments = (ins Type<MyCast_TypePred>:$newType,
                         Type<MyCast_UnaryExpressionPred>:$expr);
    let results = (outs MyCast_CastExpressionType:$res);
    let assemblyFormat = [{
        `(` $newType `:` type($newType) `)`
        `,``(` $expr `:` type($expr) `)``)` 
        attr-dict
    }];
}

def MyCast_UnopExpressionOp : MyCast_Op<"unopExpression"> {
    let arguments = (ins Type<MyCast_UnaryOperatorPred>:$op,
                         Type<MyCast_UnaryExpressionPred>:$expr);
    let results = (outs MyCast_UnopExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $op `:` type($op)`)`
        `,` `(` $expr `:` type($expr)`)``)`
        attr-dict
    }];
}

def MyCast_IncrOperatorOp : MyCast_Op<"incrOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_IncrOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_DecrOperatorOp : MyCast_Op<"decrOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_DecrOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_AddrofOperatorOp : MyCast_Op<"addrofOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_AddrofOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_DerefOperatorOp : MyCast_Op<"derefOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_DerefOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_PositiveOperatorOp : MyCast_Op<"positiveOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_PositiveOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_NegativeOperatorOp : MyCast_Op<"egativeOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_NegativeOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_NotOperatorOp : MyCast_Op<"notOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_NotOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_LnotOperatorOp : MyCast_Op<"lnotOperator"> {
    let arguments = (ins );
    let results = (outs MyCast_LnotOperatorType:$res);
    let assemblyFormat = [{
        attr-dict
    }];
}

def MyCast_SizeofExpressionOp : MyCast_Op<"sizeofExpression", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Type<MyCast_UnaryExpressionPred>>:$expr,
                         Optional<Type<MyCast_TypePred>>:$typeExpr);
    let results = (outs MyCast_SizeofExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` `(` $typeExpr `:` type($typeExpr) `)``)` 
        attr-dict
    }];
}

def MyCast_ArrayExpressionOp : MyCast_Op<"arrayExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr,
                         Type<MyCast_ExpressionPred>:$index);
    let results = (outs MyCast_ArrayExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` `(` $index `:` type($index) `)``)` 
        attr-dict
    }];
}

def MyCast_FunCallExpressionOp : MyCast_Op<"funCallExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr,
                         Variadic<Type<MyCast_AssignmentExpressionPred>>:$args);
    let results = (outs MyCast_FunCallExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` `(` $args `:` type($args) `)``)` 
        attr-dict
    }];
}

def MyCast_FieldExpressionOp : MyCast_Op<"fieldExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr,
                         Builtin_StringAttr:$field);
    let results = (outs MyCast_FieldExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` $field `)` 
        attr-dict
    }];
}

def MyCast_PtrFieldExpressionOp : MyCast_Op<"ptrFieldExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr,
                         Builtin_StringAttr:$field);
    let results = (outs MyCast_PtrFieldExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` $field `)` 
        attr-dict
    }];
}

def MyCast_PostincrExpressionOp : MyCast_Op<"postincrExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr);
    let results = (outs MyCast_PostincrExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_PostdecrExpressionOp : MyCast_Op<"postdecrExpression"> {
    let arguments = (ins Type<MyCast_PostfixExpressionPred>:$expr);
    let results = (outs MyCast_PostdecrExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_VarExpressionOp : MyCast_Op<"varExpression"> {
    let arguments = (ins Builtin_StringAttr:$name);
    let results = (outs MyCast_VarExpressionType:$res);
    let assemblyFormat = [{
        `(` $name `)`
        attr-dict
    }];
}

def MyCast_IntExpressionOp : MyCast_Op<"intExpression"> {
    let arguments = (ins Builtin_IntegerAttr:$val);
    let results = (outs MyCast_IntExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `)`
        attr-dict
    }];
}

def MyCast_FloatExpressionOp : MyCast_Op<"floatExpression"> {
    let arguments = (ins Builtin_FloatAttr:$val);
    let results = (outs MyCast_FloatExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `)`
        attr-dict
    }];
}

def MyCast_ParentExpressionOp : MyCast_Op<"parentExpression"> {
    let arguments = (ins Type<MyCast_ExpressionPred>:$expr);
    let results = (outs MyCast_ParentExpressionType:$res);
    let assemblyFormat = [{
        `(` $expr `:` type($expr) `)`
        attr-dict
    }];
}

def MyCast_StructureExpressionOp : MyCast_Op<"structureExpression"> {
    let arguments = (ins Variadic<Type<MyCast_AssignmentExpressionPred>>:$val);
    let results = (outs MyCast_StructureExpressionType:$res);
    let assemblyFormat = [{
        `(` $val `:` type($val) `)`
        attr-dict
    }];
}

def MyCast_GenericExpressionOp : MyCast_Op<"genericExpression"> {
    let arguments = (ins Type<MyCast_AssignmentExpressionPred>:$expr,
                         Variadic<Type<MyCast_GenericItemPred>>:$cases);
    let results = (outs MyCast_GenericExpressionType:$res);
    let assemblyFormat = [{
        `(``(` $expr `:` type($expr) `)`
        `,` `(` $cases `:` type($cases)`)``)`
        attr-dict
    }];
}

def MyCast_TypeGenericItemOp : MyCast_Op<"typeGenericItem"> {
    let arguments = (ins Type<MyCast_TypePred>:$cond,
                         Type<MyCast_AssignmentExpressionPred>:$body);
    let results = (outs MyCast_TypeGenericItemType:$res);
    let assemblyFormat = [{
        `(``(` $cond `:` type($cond) `)`
        `,` `(` $body `:` type($body)`)``)`
        attr-dict
    }];
}

def MyCast_DefaultGenericItemOp : MyCast_Op<"defaultGenericItem"> {
    let arguments = (ins Type<MyCast_AssignmentExpressionPred>:$body);
    let results = (outs MyCast_DefaultGenericItemType:$res);
    let assemblyFormat = [{
        `,` `(` $body `:` type($body)`)``)`
        attr-dict
    }];
}

#endif // MYCAST_OPS