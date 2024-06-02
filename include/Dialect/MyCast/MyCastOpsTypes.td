//===---------------------MyCastOpsTypes.td--------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_TYPES_TD__
#define MYCAST_TYPES_TD__

include "Dialect/MyCast/MyCastDialect.td"
include "mlir/IR/AttrTypeBase.td"

class MyCast_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<MyCast_Dialect, name, traits>
{
  let mnemonic = typeMnemonic;
}

def MyCast_ProgramType : MyCast_Type<"MyCast_Program", "program">{}
def MyCast_ProgramItemType : MyCast_Type<"MyCast_ProgramItem", "programItem">{}
def MyCast_FunDefType : MyCast_Type<"MyCast_FunDef", "funDef">{}
def MyCast_FunProtoType : MyCast_Type<"MyCast_FunProto", "funProto">{}
def MyCast_ArgumentType : MyCast_Type<"MyCast_Argument", "argument">{}
def MyCast_TypeDefType : MyCast_Type<"MyCast_TypeDef", "typeDef">{}
def MyCast_EnumDefType : MyCast_Type<"MyCast_EnumDef", "enumDef">{}
def MyCast_EnumItemType : MyCast_Type<"MyCast_EnumItem", "enumItem">{}
def MyCast_UnionDefType : MyCast_Type<"MyCast_UnionDef", "unionDef">{}
def MyCast_StructDefType : MyCast_Type<"MyCast_StructDef", "structDef">{}
def MyCast_AliasDefType : MyCast_Type<"MyCast_AliasDef", "aliasDef">{}
def MyCast_VarDeclType : MyCast_Type<"MyCast_VarDecl", "varDecl">{}
def MyCast_BaseVarDeclType : MyCast_Type<"MyCast_BaseVarDecl", "baseVarDecl">{}
def MyCast_StatementType : MyCast_Type<"MyCast_Statement", "statement">{}
def MyCast_LabeledStatementType : MyCast_Type<"MyCast_LabeledStatement", "labeledStatement">{}
def MyCast_CompoundStatementType : MyCast_Type<"MyCast_CompoundStatement", "compoundStatement">{}
def MyCast_VarDeclStatementType : MyCast_Type<"MyCast_VarDeclStatement", "varDeclStatement">{}
def MyCast_ExpressionStatementType : MyCast_Type<"MyCast_ExpressionStatement", "expressionStatement">{}
def MyCast_IfStatementType : MyCast_Type<"MyCast_IfStatement", "ifStatement">{}
def MyCast_SwitchStatementType : MyCast_Type<"MyCast_SwitchStatement", "switchStatement">{}
def MyCast_SwitchItemType : MyCast_Type<"MyCast_SwitchItem", "switchItem">{}
def MyCast_SwitchCaseItemType : MyCast_Type<"MyCast_SwitchCaseItem", "switchCaseItem">{}
def MyCast_SwitchDefaultItemType : MyCast_Type<"MyCast_SwitchDefaultItem", "switchDefaultItem">{}
def MyCast_WhileStatementType : MyCast_Type<"MyCast_WhileStatement", "whileStatement">{}
def MyCast_DoWhileStatementType : MyCast_Type<"MyCast_DoWhileStatement", "doWhileStatement">{}
def MyCast_ForStatementType : MyCast_Type<"MyCast_ForStatement", "forStatement">{}
def MyCast_GotoStatementType : MyCast_Type<"MyCast_GotoStatement", "gotoStatement">{}
def MyCast_ContinueStatementType : MyCast_Type<"MyCast_ContinueStatement", "continueStatement">{}
def MyCast_BreakStatementType : MyCast_Type<"MyCast_BreakStatement", "breakStatement">{}
def MyCast_ReturnStatementType : MyCast_Type<"MyCast_ReturnStatement", "returnStatement">{}
def MyCast_TypeType : MyCast_Type<"MyCast_Type", "type">{}
def MyCast_TypeModifierType : MyCast_Type<"MyCast_TypeModifier", "typeModifier">{}
def MyCast_StaticTypeModifierType : MyCast_Type<"MyCast_StaticTypeModifier", "staticTypeModifier">{}
def MyCast_ConstTypeModifierType : MyCast_Type<"MyCast_ConstTypeModifier", "constTypeModifier">{}
def MyCast_ExternTypeModifierType : MyCast_Type<"MyCast_ExternTypeModifier", "externTypeModifier">{}
def MyCast_VolatileTypeModifierType : MyCast_Type<"MyCast_VolatileTypeModifier", "volatileTypeModifier">{}
def MyCast_BaseTypeType : MyCast_Type<"MyCast_BaseType", "baseType">{}
def MyCast_VoidTypeType : MyCast_Type<"MyCast_VoidType", "voidType">{}
def MyCast_UnsignedLongLongTypeType : MyCast_Type<"MyCast_UnsignedLongLongType", "unsignedLongLongType">{}
def MyCast_UnsignedLongTypeType : MyCast_Type<"MyCast_UnsignedLongType", "unsignedLongType">{}
def MyCast_UnsignedShortTypeType : MyCast_Type<"MyCast_UnsignedShortType", "unsignedShortType">{}
def MyCast_UnsignedCharTypeType : MyCast_Type<"MyCast_UnsignedCharType", "unsignedCharType">{}
def MyCast_UnsignedIntTypeType : MyCast_Type<"MyCast_UnsignedIntType", "unsignedIntType">{}
def MyCast_CharTypeType : MyCast_Type<"MyCast_CharType", "charType">{}
def MyCast_ShortTypeType : MyCast_Type<"MyCast_ShortType", "shortType">{}
def MyCast_IntTypeType : MyCast_Type<"MyCast_IntType", "intType">{}
def MyCast_LongLongTypeType : MyCast_Type<"MyCast_LongLongType", "longLongType">{}
def MyCast_LongTypeType : MyCast_Type<"MyCast_LongType", "longType">{}
def MyCast_FloatTypeType : MyCast_Type<"MyCast_FloatType", "floatType">{}
def MyCast_DoubleTypeType : MyCast_Type<"MyCast_DoubleType", "doubleType">{}
def MyCast_EnumTypeType : MyCast_Type<"MyCast_EnumType", "enumType">{}
def MyCast_StructTypeType : MyCast_Type<"MyCast_StructType", "structType">{}
def MyCast_UnionTypeType : MyCast_Type<"MyCast_UnionType", "unionType">{}
def MyCast_AliasTypeType : MyCast_Type<"MyCast_AliasType", "aliasType">{}
def MyCast_ExpressionType : MyCast_Type<"MyCast_Expression", "expression">{}
def MyCast_AssignmentExpressionType : MyCast_Type<"MyCast_AssignmentExpression", "assignmentExpression">{}
def MyCast_AssignOperatorType : MyCast_Type<"MyCast_AssignOperator", "assignOperator">{}
def MyCast_EqOpType : MyCast_Type<"MyCast_EqOp", "eqOp">{}
def MyCast_StarEqOpType : MyCast_Type<"MyCast_StarEqOp", "starEqOp">{}
def MyCast_DivEqOpType : MyCast_Type<"MyCast_DivEqOp", "divEqOp">{}
def MyCast_ModuloEqOpType : MyCast_Type<"MyCast_ModuloEqOp", "moduloEqOp">{}
def MyCast_PlusEqOpType : MyCast_Type<"MyCast_PlusEqOp", "plusEqOp">{}
def MyCast_MinusEqOpType : MyCast_Type<"MyCast_MinusEqOp", "minusEqOp">{}
def MyCast_LeftShiftEqOpType : MyCast_Type<"MyCast_LeftShiftEqOp", "leftShiftEqOp">{}
def MyCast_RightShiftEqOpType : MyCast_Type<"MyCast_RightShiftEqOp", "rightShiftEqOp">{}
def MyCast_AndEqOpType : MyCast_Type<"MyCast_AndEqOp", "andEqOp">{}
def MyCast_XorEqOpType : MyCast_Type<"MyCast_XorEqOp", "xorEqOp">{}
def MyCast_OrEqOpType : MyCast_Type<"MyCast_OrEqOp", "orEqOp">{}
def MyCast_ConditionalExpressionType : MyCast_Type<"MyCast_ConditionalExpression", "conditionalExpression">{}
def MyCast_LogicalOrExpressionType : MyCast_Type<"MyCast_LogicalOrExpression", "logicalOrExpression">{}
def MyCast_LogicalAndExpressionType : MyCast_Type<"MyCast_LogicalAndExpression", "logicalAndExpression">{}
def MyCast_OrExpressionType : MyCast_Type<"MyCast_OrExpression", "orExpression">{}
def MyCast_XorExpressionType : MyCast_Type<"MyCast_XorExpression", "xorExpression">{}
def MyCast_AndExpressionType : MyCast_Type<"MyCast_AndExpression", "andExpression">{}
def MyCast_EqualityExpressionType : MyCast_Type<"MyCast_EqualityExpression", "equalityExpression">{}
def MyCast_EqualityOperatorType : MyCast_Type<"MyCast_EqualityOperator", "equalityOperator">{}
def MyCast_EqualOperatorType : MyCast_Type<"MyCast_EqualOperator", "equalOperator">{}
def MyCast_NotEqualOperatorType : MyCast_Type<"MyCast_NotEqualOperator", "notEqualOperator">{}
def MyCast_RelationalExpressionType : MyCast_Type<"MyCast_RelationalExpression", "relationalExpression">{}
def MyCast_RelationalOperatorType : MyCast_Type<"MyCast_RelationalOperator", "relationalOperator">{}
def MyCast_GeOperatorType : MyCast_Type<"MyCast_GeOperator", "geOperator">{}
def MyCast_GtOperatorType : MyCast_Type<"MyCast_GtOperator", "gtOperator">{}
def MyCast_LeOperatorType : MyCast_Type<"MyCast_LeOperator", "leOperator">{}
def MyCast_LtOperatorType : MyCast_Type<"MyCast_LtOperator", "ltOperator">{}
def MyCast_ShiftExpressionType : MyCast_Type<"MyCast_ShiftExpression", "shiftExpression">{}
def MyCast_ShiftOperatorType : MyCast_Type<"MyCast_ShiftOperator", "shiftOperator">{}
def MyCast_LshiftOperatorType : MyCast_Type<"MyCast_LshiftOperator", "lshiftOperator">{}
def MyCast_RshiftOperatorType : MyCast_Type<"MyCast_RshiftOperator", "rshiftOperator">{}
def MyCast_AdditiveExpressionType : MyCast_Type<"MyCast_AdditiveExpression", "additiveExpression">{}
def MyCast_AdditiveOperatorType : MyCast_Type<"MyCast_AdditiveOperator", "additiveOperator">{}
def MyCast_PlusOperatorType : MyCast_Type<"MyCast_PlusOperator", "plusOperator">{}
def MyCast_MinusOperatorType : MyCast_Type<"MyCast_MinusOperator", "minusOperator">{}
def MyCast_MultiplicativeExpressionType : MyCast_Type<"MyCast_MultiplicativeExpression", "multiplicativeExpression">{}
def MyCast_MultiplicativeOperatorType : MyCast_Type<"MyCast_MultiplicativeOperator", "multiplicativeOperator">{}
def MyCast_MultOperatorType : MyCast_Type<"MyCast_MultOperator", "multOperator">{}
def MyCast_DivOperatorType : MyCast_Type<"MyCast_DivOperator", "divOperator">{}
def MyCast_ModuloOperatorType : MyCast_Type<"MyCast_ModuloOperator", "moduloOperator">{}
def MyCast_CastExpressionType : MyCast_Type<"MyCast_CastExpression", "castExpression">{}
def MyCast_UnaryExpressionType : MyCast_Type<"MyCast_UnaryExpression", "unaryExpression">{}
def MyCast_UnopExpressionType : MyCast_Type<"MyCast_UnopExpression", "unopExpression">{}
def MyCast_UnaryOperatorType : MyCast_Type<"MyCast_UnaryOperator", "unaryOperator">{}
def MyCast_IncrOperatorType : MyCast_Type<"MyCast_IncrOperator", "incrOperator">{}
def MyCast_DecrOperatorType : MyCast_Type<"MyCast_DecrOperator", "decrOperator">{}
def MyCast_AddrofOperatorType : MyCast_Type<"MyCast_AddrofOperator", "addrofOperator">{}
def MyCast_DerefOperatorType : MyCast_Type<"MyCast_DerefOperator", "derefOperator">{}
def MyCast_PositiveOperatorType : MyCast_Type<"MyCast_PositiveOperator", "positiveOperator">{}
def MyCast_NegativeOperatorType : MyCast_Type<"MyCast_NegativeOperator", "negativeOperator">{}
def MyCast_NotOperatorType : MyCast_Type<"MyCast_NotOperator", "notOperator">{}
def MyCast_LnotOperatorType : MyCast_Type<"MyCast_LnotOperator", "lnotOperator">{}
def MyCast_SizeofExpressionType : MyCast_Type<"MyCast_SizeofExpression", "sizeofExpression">{}
def MyCast_PostfixExpressionType : MyCast_Type<"MyCast_PostfixExpression", "postfixExpression">{}
def MyCast_PrimaryPostfixExpressionType : MyCast_Type<"MyCast_PrimaryPostfixExpression", "primaryPostfixExpression">{}
def MyCast_ArrayExpressionType : MyCast_Type<"MyCast_ArrayExpression", "arrayExpression">{}
def MyCast_FunCallExpressionType : MyCast_Type<"MyCast_FunCallExpression", "funCallExpression">{}
def MyCast_FieldExpressionType : MyCast_Type<"MyCast_FieldExpression", "fieldExpression">{}
def MyCast_PtrFieldExpressionType : MyCast_Type<"MyCast_PtrFieldExpression", "ptrFieldExpression">{}
def MyCast_PostincrExpressionType : MyCast_Type<"MyCast_PostincrExpression", "postincrExpression">{}
def MyCast_PostdecrExpressionType : MyCast_Type<"MyCast_PostdecrExpression", "postdecrExpression">{}
def MyCast_PrimaryExpressionType : MyCast_Type<"MyCast_PrimaryExpression", "primaryExpression">{}
def MyCast_VarExpressionType : MyCast_Type<"MyCast_VarExpression", "varExpression">{}
def MyCast_IntExpressionType : MyCast_Type<"MyCast_IntExpression", "intExpression">{}
def MyCast_FloatExpressionType : MyCast_Type<"MyCast_FloatExpression", "floatExpression">{}
def MyCast_ParentExpressionType : MyCast_Type<"MyCast_ParentExpression", "parentExpression">{}
def MyCast_StructureExpressionType : MyCast_Type<"MyCast_StructureExpression", "structureExpression">{}
def MyCast_GenericExpressionType : MyCast_Type<"MyCast_GenericExpression", "genericExpression">{}
def MyCast_GenericItemType : MyCast_Type<"MyCast_GenericItem", "genericItem">{}
def MyCast_TypeGenericItemType : MyCast_Type<"MyCast_TypeGenericItem", "typeGenericItem">{}
def MyCast_DefaultGenericItemType : MyCast_Type<"MyCast_DefaultGenericItem", "defaultGenericItem">{}


def MyCast_ProgramPred : CPred<"mlir::isa<MyCast_ProgramType>($_self)">;
def MyCast_FunDefPred : CPred<"mlir::isa<MyCast_FunDefType>($_self)">;
def MyCast_FunProtoPred : CPred<"mlir::isa<MyCast_FunProtoType>($_self)">;
def MyCast_ArgumentPred : CPred<"mlir::isa<MyCast_ArgumentType>($_self)">;
def MyCast_EnumDefPred : CPred<"mlir::isa<MyCast_EnumDefType>($_self)">;
def MyCast_EnumItemPred : CPred<"mlir::isa<MyCast_EnumItemType>($_self)">;
def MyCast_UnionDefPred : CPred<"mlir::isa<MyCast_UnionDefType>($_self)">;
def MyCast_StructDefPred : CPred<"mlir::isa<MyCast_StructDefType>($_self)">;
def MyCast_AliasDefPred : CPred<"mlir::isa<MyCast_AliasDefType>($_self)">;
def MyCast_TypeDefPred : Or<[
  CPred<"mlir::isa<MyCast_TypeDefType>($_self)">,
  MyCast_UnionDefPred,
  MyCast_StructDefPred,
  MyCast_EnumDefPred,
  MyCast_AliasDefPred
]>;

def MyCast_VarDeclPred : CPred<"mlir::isa<MyCast_VarDeclType>($_self)">;
def MyCast_BaseVarDeclPred : CPred<"mlir::isa<MyCast_BaseVarDeclType>($_self)">;

def MyCast_LabeledStatementPred : CPred<"mlir::isa<MyCast_LabeledStatementType>($_self)">;
def MyCast_CompoundStatementPred : CPred<"mlir::isa<MyCast_CompoundStatementType>($_self)">;
def MyCast_VarDeclStatementPred : CPred<"mlir::isa<MyCast_VarDeclStatementType>($_self)">;
def MyCast_ExpressionStatementPred : CPred<"mlir::isa<MyCast_ExpressionStatementType>($_self)">;
def MyCast_IfStatementPred : CPred<"mlir::isa<MyCast_IfStatementType>($_self)">;
def MyCast_SwitchStatementPred : CPred<"mlir::isa<MyCast_SwitchStatementType>($_self)">;

def MyCast_SwitchCaseItemPred : CPred<"mlir::isa<MyCast_SwitchCaseItemType>($_self)">;
def MyCast_SwitchDefaultItemPred : CPred<"mlir::isa<MyCast_SwitchDefaultItemType>($_self)">;
def MyCast_SwitchItemPred : Or<[
  CPred<"mlir::isa<MyCast_SwitchItemType>($_self)">,
  MyCast_SwitchCaseItemPred,
  MyCast_SwitchDefaultItemPred
]>;
def MyCast_WhileStatementPred : CPred<"mlir::isa<MyCast_WhileStatementType>($_self)">;
def MyCast_DoWhileStatementPred : CPred<"mlir::isa<MyCast_DoWhileStatementType>($_self)">;
def MyCast_ForStatementPred : CPred<"mlir::isa<MyCast_ForStatementType>($_self)">;
def MyCast_GotoStatementPred : CPred<"mlir::isa<MyCast_GotoStatementType>($_self)">;
def MyCast_ContinueStatementPred : CPred<"mlir::isa<MyCast_ContinueStatementType>($_self)">;
def MyCast_BreakStatementPred : CPred<"mlir::isa<MyCast_BreakStatementType>($_self)">;
def MyCast_ReturnStatementPred : CPred<"mlir::isa<MyCast_ReturnStatementType>($_self)">;
def MyCast_StatementPred : Or<[
  CPred<"mlir::isa<MyCast_StatementType>($_self)">,
  MyCast_LabeledStatementPred,
  MyCast_CompoundStatementPred,
  MyCast_VarDeclStatementPred,
  MyCast_ExpressionStatementPred,
  MyCast_IfStatementPred,
  MyCast_SwitchStatementPred,
  MyCast_WhileStatementPred,
  MyCast_DoWhileStatementPred,
  MyCast_ForStatementPred,
  MyCast_GotoStatementPred,
  MyCast_ContinueStatementPred,
  MyCast_BreakStatementPred,
  MyCast_ReturnStatementPred
]>;

def MyCast_StaticTypeModifierPred : CPred<"mlir::isa<MyCast_StaticTypeModifierType>($_self)">;
def MyCast_ConstTypeModifierPred : CPred<"mlir::isa<MyCast_ConstTypeModifierType>($_self)">;
def MyCast_ExternTypeModifierPred : CPred<"mlir::isa<MyCast_ExternTypeModifierType>($_self)">;
def MyCast_VolatileTypeModifierPred : CPred<"mlir::isa<MyCast_VolatileTypeModifierType>($_self)">;
def MyCast_TypeModifierPred : Or<[
  CPred<"mlir::isa<MyCast_TypeModifierType>($_self)">,
  MyCast_StaticTypeModifierPred,
  MyCast_ConstTypeModifierPred,
  MyCast_ExternTypeModifierPred,
  MyCast_VolatileTypeModifierPred
]>;
def MyCast_VoidTypePred : CPred<"mlir::isa<MyCast_VoidTypeType>($_self)">;
def MyCast_UnsignedLongLongTypePred : CPred<"mlir::isa<MyCast_UnsignedLongLongTypeType>($_self)">;
def MyCast_UnsignedLongTypePred : CPred<"mlir::isa<MyCast_UnsignedLongTypeType>($_self)">;
def MyCast_UnsignedShortTypePred : CPred<"mlir::isa<MyCast_UnsignedShortTypeType>($_self)">;
def MyCast_UnsignedCharTypePred : CPred<"mlir::isa<MyCast_UnsignedCharTypeType>($_self)">;
def MyCast_UnsignedIntTypePred : CPred<"mlir::isa<MyCast_UnsignedIntTypeType>($_self)">;
def MyCast_CharTypePred : CPred<"mlir::isa<MyCast_CharTypeType>($_self)">;
def MyCast_ShortTypePred : CPred<"mlir::isa<MyCast_ShortTypeType>($_self)">;
def MyCast_IntTypePred : CPred<"mlir::isa<MyCast_IntTypeType>($_self)">;
def MyCast_LongLongTypePred : CPred<"mlir::isa<MyCast_LongLongTypeType>($_self)">;
def MyCast_LongTypePred : CPred<"mlir::isa<MyCast_LongTypeType>($_self)">;
def MyCast_FloatTypePred : CPred<"mlir::isa<MyCast_FloatTypeType>($_self)">;
def MyCast_DoubleTypePred : CPred<"mlir::isa<MyCast_DoubleTypeType>($_self)">;
def MyCast_EnumTypePred : CPred<"mlir::isa<MyCast_EnumTypeType>($_self)">;
def MyCast_StructTypePred : CPred<"mlir::isa<MyCast_StructTypeType>($_self)">;
def MyCast_UnionTypePred : CPred<"mlir::isa<MyCast_UnionTypeType>($_self)">;
def MyCast_AliasTypePred : CPred<"mlir::isa<MyCast_AliasTypeType>($_self)">;
def MyCast_BaseTypePred : Or<[
  CPred<"mlir::isa<MyCast_BaseTypeType>($_self)">,
  MyCast_VoidTypePred,
  MyCast_UnsignedLongLongTypePred,
  MyCast_UnsignedLongTypePred,
  MyCast_UnsignedShortTypePred,
  MyCast_UnsignedCharTypePred,
  MyCast_UnsignedIntTypePred,
  MyCast_CharTypePred,
  MyCast_ShortTypePred,
  MyCast_IntTypePred,
  MyCast_LongLongTypePred,
  MyCast_LongTypePred,
  MyCast_FloatTypePred,
  MyCast_DoubleTypePred,
  MyCast_EnumTypePred,
  MyCast_StructTypePred,
  MyCast_UnionTypePred,
  MyCast_AliasTypePred,
]>;

def MyCast_TypePred : Or<[
  CPred<"mlir::isa<MyCast_TypeType>($_self)">,
  MyCast_BaseTypePred
]>;

def MyCast_EqOpPred : CPred<"mlir::isa<MyCast_EqOpType>($_self)">;
def MyCast_StarEqOpPred : CPred<"mlir::isa<MyCast_StarEqOpType>($_self)">;
def MyCast_DivEqOpPred : CPred<"mlir::isa<MyCast_DivEqOpType>($_self)">;
def MyCast_ModuloEqOpPred : CPred<"mlir::isa<MyCast_ModuloEqOpType>($_self)">;
def MyCast_PlusEqOpPred : CPred<"mlir::isa<MyCast_PlusEqOpType>($_self)">;
def MyCast_MinusEqOpPred : CPred<"mlir::isa<MyCast_MinusEqOpType>($_self)">;
def MyCast_LeftShiftEqOpPred : CPred<"mlir::isa<MyCast_LeftShiftEqOpType>($_self)">;
def MyCast_RightShiftEqOpPred : CPred<"mlir::isa<MyCast_RightShiftEqOpType>($_self)">;
def MyCast_AndEqOpPred : CPred<"mlir::isa<MyCast_AndEqOpType>($_self)">;
def MyCast_XorEqOpPred : CPred<"mlir::isa<MyCast_XorEqOpType>($_self)">;
def MyCast_OrEqOpPred : CPred<"mlir::isa<MyCast_OrEqOpType>($_self)">;

def MyCast_AssignOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_AssignOperatorType>($_self)">,
  MyCast_EqOpPred,
  MyCast_StarEqOpPred,
  MyCast_DivEqOpPred,
  MyCast_ModuloEqOpPred,
  MyCast_PlusEqOpPred,
  MyCast_MinusEqOpPred,
  MyCast_LeftShiftEqOpPred,
  MyCast_RightShiftEqOpPred,
  MyCast_AndEqOpPred,
  MyCast_XorEqOpPred,
  MyCast_OrEqOpPred
]>;



def MyCast_EqualOperatorPred : CPred<"mlir::isa<MyCast_EqualOperatorType>($_self)">;
def MyCast_NotEqualOperatorPred : CPred<"mlir::isa<MyCast_NotEqualOperatorType>($_self)">;
def MyCast_GeOperatorPred : CPred<"mlir::isa<MyCast_GeOperatorType>($_self)">;
def MyCast_GtOperatorPred : CPred<"mlir::isa<MyCast_GtOperatorType>($_self)">;
def MyCast_LeOperatorPred : CPred<"mlir::isa<MyCast_LeOperatorType>($_self)">;
def MyCast_LtOperatorPred : CPred<"mlir::isa<MyCast_LtOperatorType>($_self)">;
def MyCast_LshiftOperatorPred : CPred<"mlir::isa<MyCast_LshiftOperatorType>($_self)">;
def MyCast_RshiftOperatorPred : CPred<"mlir::isa<MyCast_RshiftOperatorType>($_self)">;
def MyCast_PlusOperatorPred : CPred<"mlir::isa<MyCast_PlusOperatorType>($_self)">;
def MyCast_MinusOperatorPred : CPred<"mlir::isa<MyCast_MinusOperatorType>($_self)">;
def MyCast_MultOperatorPred : CPred<"mlir::isa<MyCast_MultOperatorType>($_self)">;
def MyCast_DivOperatorPred : CPred<"mlir::isa<MyCast_DivOperatorType>($_self)">;
def MyCast_ModuloOperatorPred : CPred<"mlir::isa<MyCast_ModuloOperatorType>($_self)">;
def MyCast_UnopExpressionPred : CPred<"mlir::isa<MyCast_UnopExpressionType>($_self)">;
def MyCast_IncrOperatorPred : CPred<"mlir::isa<MyCast_IncrOperatorType>($_self)">;
def MyCast_DecrOperatorPred : CPred<"mlir::isa<MyCast_DecrOperatorType>($_self)">;
def MyCast_AddrofOperatorPred : CPred<"mlir::isa<MyCast_AddrofOperatorType>($_self)">;
def MyCast_DerefOperatorPred : CPred<"mlir::isa<MyCast_DerefOperatorType>($_self)">;
def MyCast_PositiveOperatorPred : CPred<"mlir::isa<MyCast_PositiveOperatorType>($_self)">;
def MyCast_NegativeOperatorPred : CPred<"mlir::isa<MyCast_NegativeOperatorType>($_self)">;
def MyCast_NotOperatorPred : CPred<"mlir::isa<MyCast_NotOperatorType>($_self)">;
def MyCast_LnotOperatorPred : CPred<"mlir::isa<MyCast_LnotOperatorType>($_self)">;
def MyCast_SizeofExpressionPred : CPred<"mlir::isa<MyCast_SizeofExpressionType>($_self)">;

def MyCast_ArrayExpressionPred : CPred<"mlir::isa<MyCast_ArrayExpressionType>($_self)">;
def MyCast_FunCallExpressionPred : CPred<"mlir::isa<MyCast_FunCallExpressionType>($_self)">;
def MyCast_FieldExpressionPred : CPred<"mlir::isa<MyCast_FieldExpressionType>($_self)">;
def MyCast_PtrFieldExpressionPred : CPred<"mlir::isa<MyCast_PtrFieldExpressionType>($_self)">;
def MyCast_PostincrExpressionPred : CPred<"mlir::isa<MyCast_PostincrExpressionType>($_self)">;
def MyCast_PostdecrExpressionPred : CPred<"mlir::isa<MyCast_PostdecrExpressionType>($_self)">;
def MyCast_VarExpressionPred : CPred<"mlir::isa<MyCast_VarExpressionType>($_self)">;
def MyCast_IntExpressionPred : CPred<"mlir::isa<MyCast_IntExpressionType>($_self)">;
def MyCast_FloatExpressionPred : CPred<"mlir::isa<MyCast_FloatExpressionType>($_self)">;
def MyCast_ParentExpressionPred : CPred<"mlir::isa<MyCast_ParentExpressionType>($_self)">;
def MyCast_StructureExpressionPred : CPred<"mlir::isa<MyCast_StructureExpressionType>($_self)">;
def MyCast_GenericExpressionPred : CPred<"mlir::isa<MyCast_GenericExpressionType>($_self)">;
def MyCast_TypeGenericItemPred : CPred<"mlir::isa<MyCast_TypeGenericItemType>($_self)">;
def MyCast_DefaultGenericItemPred : CPred<"mlir::isa<MyCast_DefaultGenericItemType>($_self)">;


def MyCast_GenericItemPred : Or<[
  CPred<"mlir::isa<MyCast_GenericItemType>($_self)">,
  MyCast_TypeGenericItemPred,
  MyCast_DefaultGenericItemPred
]>;

def MyCast_PrimaryExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_PrimaryExpressionType>($_self)">,
  MyCast_VarExpressionPred,
  MyCast_IntExpressionPred,
  MyCast_FloatExpressionPred,
  MyCast_ParentExpressionPred,
  MyCast_StructureExpressionPred,
  MyCast_GenericExpressionPred
]>;


def MyCast_PrimaryPostfixExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_PrimaryPostfixExpressionType>($_self)">,
  MyCast_PrimaryExpressionPred
]>;

def MyCast_PostfixExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_PostfixExpressionType>($_self)">,
  MyCast_PrimaryPostfixExpressionPred,
  MyCast_ArrayExpressionPred,
  MyCast_FunCallExpressionPred,
  MyCast_FieldExpressionPred,
  MyCast_PtrFieldExpressionPred,
  MyCast_PostincrExpressionPred,
  MyCast_PostdecrExpressionPred
]>;

def MyCast_UnaryOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_UnaryOperatorType>($_self)">,
  MyCast_IncrOperatorPred,
  MyCast_DecrOperatorPred,
  MyCast_AddrofOperatorPred,
  MyCast_DerefOperatorPred,
  MyCast_PositiveOperatorPred,
  MyCast_NegativeOperatorPred,
  MyCast_NotOperatorPred,
  MyCast_LnotOperatorPred
]>;

def MyCast_UnaryExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_UnaryExpressionType>($_self)">,
  MyCast_PostfixExpressionPred,
  MyCast_SizeofExpressionPred,
  MyCast_UnopExpressionPred
]>;

def MyCast_CastExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_CastExpressionType>($_self)">,
  MyCast_UnaryExpressionPred
]>;

def MyCast_MultiplicativeOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_MultiplicativeOperatorType>($_self)">,
  MyCast_MultOperatorPred,
  MyCast_DivOperatorPred,
  MyCast_ModuloOperatorPred
]>;

def MyCast_MultiplicativeExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_MultiplicativeExpressionType>($_self)">,
  MyCast_CastExpressionPred
]>;

def MyCast_AdditiveOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_AdditiveOperatorType>($_self)">,
  MyCast_PlusOperatorPred,
  MyCast_MinusOperatorPred
]>;

def MyCast_AdditiveExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_AdditiveExpressionType>($_self)">,
  MyCast_MultiplicativeExpressionPred
]>;

def MyCast_ShiftOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_ShiftOperatorType>($_self)">,
  MyCast_LshiftOperatorPred,
  MyCast_RshiftOperatorPred
]>;

def MyCast_ShiftExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_ShiftExpressionType>($_self)">,
  MyCast_AdditiveExpressionPred
]>;

def MyCast_RelationalOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_RelationalOperatorType>($_self)">,
  MyCast_GeOperatorPred,
  MyCast_GtOperatorPred,
  MyCast_LeOperatorPred,
  MyCast_LtOperatorPred
]>;

def MyCast_RelationalExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_RelationalExpressionType>($_self)">,
  MyCast_ShiftExpressionPred
]>;


def MyCast_EqualityOperatorPred : Or<[
  CPred<"mlir::isa<MyCast_EqualityOperatorType>($_self)">,
  MyCast_EqualOperatorPred,
  MyCast_NotEqualOperatorPred
]>;

def MyCast_EqualityExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_EqualityExpressionType>($_self)">,
  MyCast_RelationalExpressionPred
]>;

def MyCast_AndExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_AndExpressionType>($_self)">,
  MyCast_EqualityExpressionPred
]>;

def MyCast_XorExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_XorExpressionType>($_self)">,
  MyCast_AndExpressionPred
]>;

def MyCast_OrExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_OrExpressionType>($_self)">,
  MyCast_XorExpressionPred
]>;

def MyCast_LogicalAndExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_LogicalAndExpressionType>($_self)">,
  MyCast_OrExpressionPred
]>;

def MyCast_LogicalOrExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_LogicalOrExpressionType>($_self)">,
  MyCast_LogicalAndExpressionPred
]>;


def MyCast_ConditionalExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_ConditionalExpressionType>($_self)">,
  MyCast_LogicalOrExpressionPred
]>;


def MyCast_AssignmentExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_AssignmentExpressionType>($_self)">,
  MyCast_ConditionalExpressionPred
]>;
def MyCast_ExpressionPred : Or<[
  CPred<"mlir::isa<MyCast_ExpressionType>($_self)">,
  MyCast_AssignmentExpressionPred
]>;

def MyCast_ProgramItemPred : Or<[
  CPred<"mlir::isa<MyCast_ProgramItemType>($_self)">,
  MyCast_FunDefPred,
  MyCast_VarDeclPred,
  MyCast_TypeDefPred
]>;

#endif // MYCAST_TYPES_TD__
