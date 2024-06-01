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


def MyCast_ProgramPred : CPred<"$_self.isa<MyCast_ProgramType>()">;
def MyCast_FunDefPred : CPred<"$_self.isa<MyCast_FunDefType>()">;
def MyCast_FunProtoPred : CPred<"$_self.isa<MyCast_FunProtoType>()">;
def MyCast_ArgumentPred : CPred<"$_self.isa<MyCast_ArgumentType>()">;
def MyCast_EnumDefPred : CPred<"$_self.isa<MyCast_EnumDefType>()">;
def MyCast_EnumItemPred : CPred<"$_self.isa<MyCast_EnumItemType>()">;
def MyCast_UnionDefPred : CPred<"$_self.isa<MyCast_UnionDefType>()">;
def MyCast_StructDefPred : CPred<"$_self.isa<MyCast_StructDefType>()">;
def MyCast_AliasDefPred : CPred<"$_self.isa<MyCast_AliasDefType>()">;
def MyCast_TypeDefPred : Or<[
  CPred<"$_self.isa<MyCast_TypeDefType>()">,
  MyCast_UnionDefPred,
  MyCast_StructDefPred,
  MyCast_EnumDefPred,
  MyCast_AliasDefPred
]>;

def MyCast_VarDeclPred : CPred<"$_self.isa<MyCast_VarDeclType>()">;
def MyCast_BaseVarDeclPred : CPred<"$_self.isa<MyCast_BaseVarDeclType>()">;

def MyCast_LabeledStatementPred : CPred<"$_self.isa<MyCast_LabeledStatementType>()">;
def MyCast_CompoundStatementPred : CPred<"$_self.isa<MyCast_CompoundStatementType>()">;
def MyCast_VarDeclStatementPred : CPred<"$_self.isa<MyCast_VarDeclStatementType>()">;
def MyCast_ExpressionStatementPred : CPred<"$_self.isa<MyCast_ExpressionStatementType>()">;
def MyCast_IfStatementPred : CPred<"$_self.isa<MyCast_IfStatementType>()">;
def MyCast_SwitchStatementPred : CPred<"$_self.isa<MyCast_SwitchStatementType>()">;

def MyCast_SwitchCaseItemPred : CPred<"$_self.isa<MyCast_SwitchCaseItemType>()">;
def MyCast_SwitchDefaultItemPred : CPred<"$_self.isa<MyCast_SwitchDefaultItemType>()">;
def MyCast_SwitchItemPred : Or<[
  CPred<"$_self.isa<MyCast_SwitchItemType>()">,
  MyCast_SwitchCaseItemPred,
  MyCast_SwitchDefaultItemPred
]>;
def MyCast_WhileStatementPred : CPred<"$_self.isa<MyCast_WhileStatementType>()">;
def MyCast_DoWhileStatementPred : CPred<"$_self.isa<MyCast_DoWhileStatementType>()">;
def MyCast_ForStatementPred : CPred<"$_self.isa<MyCast_ForStatementType>()">;
def MyCast_GotoStatementPred : CPred<"$_self.isa<MyCast_GotoStatementType>()">;
def MyCast_ContinueStatementPred : CPred<"$_self.isa<MyCast_ContinueStatementType>()">;
def MyCast_BreakStatementPred : CPred<"$_self.isa<MyCast_BreakStatementType>()">;
def MyCast_ReturnStatementPred : CPred<"$_self.isa<MyCast_ReturnStatementType>()">;
def MyCast_StatementPred : Or<[
  CPred<"$_self.isa<MyCast_StatementType>()">,
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

def MyCast_StaticTypeModifierPred : CPred<"$_self.isa<MyCast_StaticTypeModifierType>()">;
def MyCast_ConstTypeModifierPred : CPred<"$_self.isa<MyCast_ConstTypeModifierType>()">;
def MyCast_ExternTypeModifierPred : CPred<"$_self.isa<MyCast_ExternTypeModifierType>()">;
def MyCast_VolatileTypeModifierPred : CPred<"$_self.isa<MyCast_VolatileTypeModifierType>()">;
def MyCast_TypeModifierPred : Or<[
  CPred<"$_self.isa<MyCast_TypeModifierType>()">,
  MyCast_StaticTypeModifierPred,
  MyCast_ConstTypeModifierPred,
  MyCast_ExternTypeModifierPred,
  MyCast_VolatileTypeModifierPred
]>;
def MyCast_VoidTypePred : CPred<"$_self.isa<MyCast_VoidTypeType>()">;
def MyCast_UnsignedLongLongTypePred : CPred<"$_self.isa<MyCast_UnsignedLongLongTypeType>()">;
def MyCast_UnsignedLongTypePred : CPred<"$_self.isa<MyCast_UnsignedLongTypeType>()">;
def MyCast_UnsignedShortTypePred : CPred<"$_self.isa<MyCast_UnsignedShortTypeType>()">;
def MyCast_UnsignedCharTypePred : CPred<"$_self.isa<MyCast_UnsignedCharTypeType>()">;
def MyCast_UnsignedIntTypePred : CPred<"$_self.isa<MyCast_UnsignedIntTypeType>()">;
def MyCast_CharTypePred : CPred<"$_self.isa<MyCast_CharTypeType>()">;
def MyCast_ShortTypePred : CPred<"$_self.isa<MyCast_ShortTypeType>()">;
def MyCast_IntTypePred : CPred<"$_self.isa<MyCast_IntTypeType>()">;
def MyCast_LongLongTypePred : CPred<"$_self.isa<MyCast_LongLongTypeType>()">;
def MyCast_LongTypePred : CPred<"$_self.isa<MyCast_LongTypeType>()">;
def MyCast_FloatTypePred : CPred<"$_self.isa<MyCast_FloatTypeType>()">;
def MyCast_DoubleTypePred : CPred<"$_self.isa<MyCast_DoubleTypeType>()">;
def MyCast_EnumTypePred : CPred<"$_self.isa<MyCast_EnumTypeType>()">;
def MyCast_StructTypePred : CPred<"$_self.isa<MyCast_StructTypeType>()">;
def MyCast_UnionTypePred : CPred<"$_self.isa<MyCast_UnionTypeType>()">;
def MyCast_AliasTypePred : CPred<"$_self.isa<MyCast_AliasTypeType>()">;
def MyCast_BaseTypePred : Or<[
  CPred<"$_self.isa<MyCast_BaseTypeType>()">,
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
  CPred<"$_self.isa<MyCast_TypeType>()">,
  MyCast_BaseTypePred
]>;

def MyCast_EqOpPred : CPred<"$_self.isa<MyCast_EqOpType>()">;
def MyCast_StarEqOpPred : CPred<"$_self.isa<MyCast_StarEqOpType>()">;
def MyCast_DivEqOpPred : CPred<"$_self.isa<MyCast_DivEqOpType>()">;
def MyCast_ModuloEqOpPred : CPred<"$_self.isa<MyCast_ModuloEqOpType>()">;
def MyCast_PlusEqOpPred : CPred<"$_self.isa<MyCast_PlusEqOpType>()">;
def MyCast_MinusEqOpPred : CPred<"$_self.isa<MyCast_MinusEqOpType>()">;
def MyCast_LeftShiftEqOpPred : CPred<"$_self.isa<MyCast_LeftShiftEqOpType>()">;
def MyCast_RightShiftEqOpPred : CPred<"$_self.isa<MyCast_RightShiftEqOpType>()">;
def MyCast_AndEqOpPred : CPred<"$_self.isa<MyCast_AndEqOpType>()">;
def MyCast_XorEqOpPred : CPred<"$_self.isa<MyCast_XorEqOpType>()">;
def MyCast_OrEqOpPred : CPred<"$_self.isa<MyCast_OrEqOpType>()">;

def MyCast_AssignOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_AssignOperatorType>()">,
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



def MyCast_EqualOperatorPred : CPred<"$_self.isa<MyCast_EqualOperatorType>()">;
def MyCast_NotEqualOperatorPred : CPred<"$_self.isa<MyCast_NotEqualOperatorType>()">;
def MyCast_GeOperatorPred : CPred<"$_self.isa<MyCast_GeOperatorType>()">;
def MyCast_GtOperatorPred : CPred<"$_self.isa<MyCast_GtOperatorType>()">;
def MyCast_LeOperatorPred : CPred<"$_self.isa<MyCast_LeOperatorType>()">;
def MyCast_LtOperatorPred : CPred<"$_self.isa<MyCast_LtOperatorType>()">;
def MyCast_LshiftOperatorPred : CPred<"$_self.isa<MyCast_LshiftOperatorType>()">;
def MyCast_RshiftOperatorPred : CPred<"$_self.isa<MyCast_RshiftOperatorType>()">;
def MyCast_PlusOperatorPred : CPred<"$_self.isa<MyCast_PlusOperatorType>()">;
def MyCast_MinusOperatorPred : CPred<"$_self.isa<MyCast_MinusOperatorType>()">;
def MyCast_MultOperatorPred : CPred<"$_self.isa<MyCast_MultOperatorType>()">;
def MyCast_DivOperatorPred : CPred<"$_self.isa<MyCast_DivOperatorType>()">;
def MyCast_ModuloOperatorPred : CPred<"$_self.isa<MyCast_ModuloOperatorType>()">;
def MyCast_UnopExpressionPred : CPred<"$_self.isa<MyCast_UnopExpressionType>()">;
def MyCast_IncrOperatorPred : CPred<"$_self.isa<MyCast_IncrOperatorType>()">;
def MyCast_DecrOperatorPred : CPred<"$_self.isa<MyCast_DecrOperatorType>()">;
def MyCast_AddrofOperatorPred : CPred<"$_self.isa<MyCast_AddrofOperatorType>()">;
def MyCast_DerefOperatorPred : CPred<"$_self.isa<MyCast_DerefOperatorType>()">;
def MyCast_PositiveOperatorPred : CPred<"$_self.isa<MyCast_PositiveOperatorType>()">;
def MyCast_NegativeOperatorPred : CPred<"$_self.isa<MyCast_NegativeOperatorType>()">;
def MyCast_NotOperatorPred : CPred<"$_self.isa<MyCast_NotOperatorType>()">;
def MyCast_LnotOperatorPred : CPred<"$_self.isa<MyCast_LnotOperatorType>()">;
def MyCast_SizeofExpressionPred : CPred<"$_self.isa<MyCast_SizeofExpressionType>()">;

def MyCast_ArrayExpressionPred : CPred<"$_self.isa<MyCast_ArrayExpressionType>()">;
def MyCast_FunCallExpressionPred : CPred<"$_self.isa<MyCast_FunCallExpressionType>()">;
def MyCast_FieldExpressionPred : CPred<"$_self.isa<MyCast_FieldExpressionType>()">;
def MyCast_PtrFieldExpressionPred : CPred<"$_self.isa<MyCast_PtrFieldExpressionType>()">;
def MyCast_PostincrExpressionPred : CPred<"$_self.isa<MyCast_PostincrExpressionType>()">;
def MyCast_PostdecrExpressionPred : CPred<"$_self.isa<MyCast_PostdecrExpressionType>()">;
def MyCast_VarExpressionPred : CPred<"$_self.isa<MyCast_VarExpressionType>()">;
def MyCast_IntExpressionPred : CPred<"$_self.isa<MyCast_IntExpressionType>()">;
def MyCast_FloatExpressionPred : CPred<"$_self.isa<MyCast_FloatExpressionType>()">;
def MyCast_ParentExpressionPred : CPred<"$_self.isa<MyCast_ParentExpressionType>()">;
def MyCast_StructureExpressionPred : CPred<"$_self.isa<MyCast_StructureExpressionType>()">;
def MyCast_GenericExpressionPred : CPred<"$_self.isa<MyCast_GenericExpressionType>()">;
def MyCast_TypeGenericItemPred : CPred<"$_self.isa<MyCast_TypeGenericItemType>()">;
def MyCast_DefaultGenericItemPred : CPred<"$_self.isa<MyCast_DefaultGenericItemType>()">;


def MyCast_GenericItemPred : Or<[
  CPred<"$_self.isa<MyCast_GenericItemType>()">,
  MyCast_TypeGenericItemPred,
  MyCast_DefaultGenericItemPred
]>;

def MyCast_PrimaryExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_PrimaryExpressionType>()">,
  MyCast_VarExpressionPred,
  MyCast_IntExpressionPred,
  MyCast_FloatExpressionPred,
  MyCast_ParentExpressionPred,
  MyCast_StructureExpressionPred,
  MyCast_GenericExpressionPred
]>;


def MyCast_PrimaryPostfixExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_PrimaryPostfixExpressionType>()">,
  MyCast_PrimaryExpressionPred
]>;

def MyCast_PostfixExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_PostfixExpressionType>()">,
  MyCast_PrimaryPostfixExpressionPred,
  MyCast_ArrayExpressionPred,
  MyCast_FunCallExpressionPred,
  MyCast_FieldExpressionPred,
  MyCast_PtrFieldExpressionPred,
  MyCast_PostincrExpressionPred,
  MyCast_PostdecrExpressionPred
]>;

def MyCast_UnaryOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_UnaryOperatorType>()">,
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
  CPred<"$_self.isa<MyCast_UnaryExpressionType>()">,
  MyCast_PostfixExpressionPred,
  MyCast_SizeofExpressionPred,
  MyCast_UnopExpressionPred
]>;

def MyCast_CastExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_CastExpressionType>()">,
  MyCast_UnaryExpressionPred
]>;

def MyCast_MultiplicativeOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_MultiplicativeOperatorType>()">,
  MyCast_MultOperatorPred,
  MyCast_DivOperatorPred,
  MyCast_ModuloOperatorPred
]>;

def MyCast_MultiplicativeExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_MultiplicativeExpressionType>()">,
  MyCast_CastExpressionPred
]>;

def MyCast_AdditiveOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_AdditiveOperatorType>()">,
  MyCast_PlusOperatorPred,
  MyCast_MinusOperatorPred
]>;

def MyCast_AdditiveExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_AdditiveExpressionType>()">,
  MyCast_MultiplicativeExpressionPred
]>;

def MyCast_ShiftOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_ShiftOperatorType>()">,
  MyCast_LshiftOperatorPred,
  MyCast_RshiftOperatorPred
]>;

def MyCast_ShiftExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_ShiftExpressionType>()">,
  MyCast_AdditiveExpressionPred
]>;

def MyCast_RelationalOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_RelationalOperatorType>()">,
  MyCast_GeOperatorPred,
  MyCast_GtOperatorPred,
  MyCast_LeOperatorPred,
  MyCast_LtOperatorPred
]>;

def MyCast_RelationalExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_RelationalExpressionType>()">,
  MyCast_ShiftExpressionPred
]>;


def MyCast_EqualityOperatorPred : Or<[
  CPred<"$_self.isa<MyCast_EqualityOperatorType>()">,
  MyCast_EqualOperatorPred,
  MyCast_NotEqualOperatorPred
]>;

def MyCast_EqualityExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_EqualityExpressionType>()">,
  MyCast_RelationalExpressionPred
]>;

def MyCast_AndExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_AndExpressionType>()">,
  MyCast_EqualityExpressionPred
]>;

def MyCast_XorExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_XorExpressionType>()">,
  MyCast_AndExpressionPred
]>;

def MyCast_OrExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_OrExpressionType>()">,
  MyCast_XorExpressionPred
]>;

def MyCast_LogicalAndExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_LogicalAndExpressionType>()">,
  MyCast_OrExpressionPred
]>;

def MyCast_LogicalOrExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_LogicalOrExpressionType>()">,
  MyCast_LogicalAndExpressionPred
]>;


def MyCast_ConditionalExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_ConditionalExpressionType>()">,
  MyCast_LogicalOrExpressionPred
]>;


def MyCast_AssignmentExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_AssignmentExpressionType>()">,
  MyCast_ConditionalExpressionPred
]>;
def MyCast_ExpressionPred : Or<[
  CPred<"$_self.isa<MyCast_ExpressionType>()">,
  MyCast_AssignmentExpressionPred
]>;

#endif // MYCAST_TYPES_TD__
