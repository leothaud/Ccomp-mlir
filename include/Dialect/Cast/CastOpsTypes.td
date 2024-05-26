//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CAST_TYPES_TD__
#define CAST_TYPES_TD__

include "Dialect/Cast/CastDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Cast_AstNodeType<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<Cast_Dialect, name, traits>
{
  let mnemonic = typeMnemonic;
}


def Cast_ProgramNodeType : Cast_AstNodeType<"Cast_ProgramNode", "programNode">{}
def Cast_FunDefNodeType : Cast_AstNodeType<"Cast_FunDefNode", "funDefNode">{}
def Cast_FunProtoNodeType : Cast_AstNodeType<"Cast_FunProtoNode", "funProtoNode">{}
def Cast_ArgumentNodeType : Cast_AstNodeType<"Cast_ArgumentNode", "argumentNode">{}
def Cast_VarDeclNodeType : Cast_AstNodeType<"Cast_VarDeclNode", "varDeclNode">{}
def Cast_BaseVarDeclNodeType : Cast_AstNodeType<"Cast_BaseVarDeclNode", "baseVarDeclNode">{}
def Cast_UnionDefNodeType : Cast_AstNodeType<"Cast_UnionDefNode", "unionDefNode">{}
def Cast_StructDefNodeType : Cast_AstNodeType<"Cast_StructDefNode", "structDefNode">{}
def Cast_EnumDefNodeType : Cast_AstNodeType<"Cast_EnumDefNode", "enumDefNode">{}
def Cast_AliasDefNodeType : Cast_AstNodeType<"Cast_AliasDefNode", "aliasDefNode">{}
def Cast_EnumItemNodeType : Cast_AstNodeType<"Cast_EnumItemNode", "enumItemNode">{}
def Cast_LabeledStatementNodeType : Cast_AstNodeType<"Cast_LabeledStatementNode", "labeledStatementNode">{}
def Cast_CompoundStatementNodeType : Cast_AstNodeType<"Cast_CompoundStatementNode", "compoundStatementNode">{}
def Cast_VarDeclStatementNodeType : Cast_AstNodeType<"Cast_VarDeclStatementNode", "varDeclStatementNode">{}
def Cast_ExpressionStatementNodeType : Cast_AstNodeType<"Cast_ExpressionStatementNode", "expressionStatementNode">{}
def Cast_IfStatementNodeType : Cast_AstNodeType<"Cast_IfStatementNode", "ifStatementNode">{}
def Cast_SwitchStatementNodeType : Cast_AstNodeType<"Cast_SwitchStatementNode", "switchStatementNode">{}
def Cast_SwitchCaseItemNodeType : Cast_AstNodeType<"Cast_SwitchCaseItemNode", "switchCaseItemNode">{}
def Cast_SwitchDefaultItemNodeType : Cast_AstNodeType<"Cast_SwitchDefaultItemNode", "switchDefaultItemNode">{}
def Cast_WhileStatementNodeType : Cast_AstNodeType<"Cast_WhileStatementNode", "whileStatementNode">{}
def Cast_DoWhileStatementNodeType : Cast_AstNodeType<"Cast_DoWhileStatementNode", "doWhileStatementNode">{}
def Cast_ForStatementNodeType : Cast_AstNodeType<"Cast_ForStatementNode", "forStatementNode">{}
def Cast_GotoStatementNodeType : Cast_AstNodeType<"Cast_GotoStatementNode", "gotoStatementNode">{}
def Cast_ContinueStatementNodeType : Cast_AstNodeType<"Cast_ContinueStatementNode", "continueStatementNode">{}
def Cast_BreakStatementNodeType : Cast_AstNodeType<"Cast_BreakStatementNode", "breakStatementNode">{}
def Cast_ReturnStatementNodeType : Cast_AstNodeType<"Cast_ReturnStatementNode", "returnStatementNode">{}
def Cast_TypeNodeType : Cast_AstNodeType<"Cast_TypeNode", "typeNode">{}
def Cast_ExternTypeModifierNodeType : Cast_AstNodeType<"Cast_ExternTypeModifierNode", "externTypeModifierNode">{}
def Cast_StaticTypeModifierNodeType : Cast_AstNodeType<"Cast_StaticTypeModifierNode", "staticTypeModifierNode">{}
def Cast_RegisterTypeModifierNodeType : Cast_AstNodeType<"Cast_RegisterTypeModifierNode", "registerTypeModifierNode">{}
def Cast_ConstTypeModifierNodeType : Cast_AstNodeType<"Cast_ConstTypeModifierNode", "constTypeModifierNode">{}
def Cast_VolatileTypeModifierNodeType : Cast_AstNodeType<"Cast_VolatileTypeModifierNode", "volatileTypeModifierNode">{}
def Cast_VoidTypeNodeType : Cast_AstNodeType<"Cast_VoidTypeNode", "voidTypeNode">{}
def Cast_UnsignedLongLongTypeNodeType : Cast_AstNodeType<"Cast_UnsignedLongLongTypeNode", "unsignedLongLongTypeNode">{}
def Cast_UnsignedLongTypeNodeType : Cast_AstNodeType<"Cast_UnsignedLongTypeNode", "unsignedLongTypeNode">{}
def Cast_UnsignedShortTypeNodeType : Cast_AstNodeType<"Cast_UnsignedShortTypeNode", "unsignedShortTypeNode">{}
def Cast_UnsignedCharTypeNodeType : Cast_AstNodeType<"Cast_UnsignedCharTypeNode", "unsignedCharTypeNode">{}
def Cast_UnsignedIntTypeNodeType : Cast_AstNodeType<"Cast_UnsignedIntTypeNode", "unsignedIntTypeNode">{}
def Cast_CharTypeNodeType : Cast_AstNodeType<"Cast_CharTypeNode", "charTypeNode">{}
def Cast_ShortTypeNodeType : Cast_AstNodeType<"Cast_ShortTypeNode", "shortTypeNode">{}
def Cast_IntTypeNodeType : Cast_AstNodeType<"Cast_IntTypeNode", "intTypeNode">{}
def Cast_LongLongTypeNodeType : Cast_AstNodeType<"Cast_LongLongTypeNode", "longLongTypeNode">{}
def Cast_LongTypeNodeType : Cast_AstNodeType<"Cast_LongTypeNode", "longTypeNode">{}
def Cast_FloatTypeNodeType : Cast_AstNodeType<"Cast_FloatTypeNode", "floatTypeNode">{}
def Cast_DoubleTypeNodeType : Cast_AstNodeType<"Cast_DoubleTypeNode", "doubleTypeNode">{}
def Cast_EnumTypeNodeType : Cast_AstNodeType<"Cast_EnumTypeNode", "enumTypeNode">{}
def Cast_StructTypeNodeType : Cast_AstNodeType<"Cast_StructTypeNode", "structTypeNode">{}
def Cast_UnionTypeNodeType : Cast_AstNodeType<"Cast_UnionTypeNode", "unionTypeNode">{}
def Cast_AliasTypeNodeType : Cast_AstNodeType<"Cast_AliasTypeNode", "aliasTypeNode">{}
def Cast_ExpressionNodeType : Cast_AstNodeType<"Cast_ExpressionNode", "expressionNode">{}
def Cast_AssignOpExpressionNodeType : Cast_AstNodeType<"Cast_AssignOpExpressionNode", "assignOpExpressionNode">{}
def Cast_EqOpNodeType : Cast_AstNodeType<"Cast_EqOpNode", "EqOpNode">{}
def Cast_StarEqOpNodeType : Cast_AstNodeType<"Cast_StarEqOpNode", "starEqOpNode">{}
def Cast_DivEqOpNodeType : Cast_AstNodeType<"Cast_DivEqOpNode", "divEqOpNode">{}
def Cast_ModuloEqOpNodeType : Cast_AstNodeType<"Cast_ModuloEqOpNode", "moduloEqOpNode">{}
def Cast_PlusEqOpNodeType : Cast_AstNodeType<"Cast_PlusEqOpNode", "plusEqOpNode">{}
def Cast_MinusEqOpNodeType : Cast_AstNodeType<"Cast_MinusEqOpNode", "minusEqOpNode">{}
def Cast_LeftShiftEqOpNodeType : Cast_AstNodeType<"Cast_LeftShiftEqOpNode", "leftShiftEqOpNode">{}
def Cast_RightShiftEqOpNodeType : Cast_AstNodeType<"Cast_RightShiftEqOpNode", "rightShiftEqOpNode">{}
def Cast_AndEqOpNodeType : Cast_AstNodeType<"Cast_AndEqOpNode", "andEqOpNode">{}
def Cast_XorEqOpNodeType : Cast_AstNodeType<"Cast_XorEqOpNode", "xorEqOpNode">{}
def Cast_OrEqOpNodeType : Cast_AstNodeType<"Cast_OrEqOpNode", "orEqOpNode">{}
def Cast_ConditionalOpExpressionNodeType : Cast_AstNodeType<"Cast_ConditionalOpExpressionNode", "conditionalOpExpressionNode">{}
def Cast_LogicalOrExpressionNodeType : Cast_AstNodeType<"Cast_LogicalOrExpressionNode", "logicalOrExpressionNode">{}
def Cast_LogicalAndExpressionNodeType : Cast_AstNodeType<"Cast_LogicalAndExpressionNode", "logicalAndExpressionNode">{}
def Cast_OrExpressionNodeType : Cast_AstNodeType<"Cast_OrExpressionNode", "orExpressionNode">{}
def Cast_XorExpressionNodeType : Cast_AstNodeType<"Cast_XorExpressionNode", "xorExpressionNode">{}
def Cast_AndExpressionNodeType : Cast_AstNodeType<"Cast_AndExpressionNode", "andExpressionNode">{}
def Cast_EqualOpExpressionNodeType : Cast_AstNodeType<"Cast_EqualOpExpressionNode", "equalOpExpressionNode">{}
def Cast_EqualOperatorNodeType : Cast_AstNodeType<"Cast_EqualOperatorNode", "equalOperatorNode">{}
def Cast_NotEqualOperatorNodeType : Cast_AstNodeType<"Cast_NotEqualOperatorNode", "notEqualOperatorNode">{}
def Cast_RelOpExpressionNodeType : Cast_AstNodeType<"Cast_RelOpExpressionNode", "relOpExpressionNode">{}
def Cast_GeOperatorNodeType : Cast_AstNodeType<"Cast_GeOperatorNode", "geOperatorNode">{}
def Cast_GtOperatorNodeType : Cast_AstNodeType<"Cast_GtOperatorNode", "gtOperatorNode">{}
def Cast_LeOperatorNodeType : Cast_AstNodeType<"Cast_LeOperatorNode", "leOperatorNode">{}
def Cast_LtOperatorNodeType : Cast_AstNodeType<"Cast_LtOperatorNode", "ltOperatorNode">{}
def Cast_ShiftOpExpressionNodeType : Cast_AstNodeType<"Cast_ShiftOpExpressionNode", "shiftOpExpressionNode">{}
def Cast_LshiftOperatorNodeType : Cast_AstNodeType<"Cast_LshiftOperatorNode", "lshiftOperatorNode">{}
def Cast_RshiftOperatorNodeType : Cast_AstNodeType<"Cast_RshiftOperatorNode", "rshiftOperatorNode">{}
def Cast_AddOpExpressionNodeType : Cast_AstNodeType<"Cast_AddOpExpressionNode", "addOpExpressionNode">{}
def Cast_PlusOperatorNodeType : Cast_AstNodeType<"Cast_PlusOperatorNode", "plusOperatorNode">{}
def Cast_MinusOperatorNodeType : Cast_AstNodeType<"Cast_MinusOperatorNode", "minusOperatorNode">{}
def Cast_MulOpExpressionNodeType : Cast_AstNodeType<"Cast_MulOpExpressionNode", "mulOpExpressionNode">{}
def Cast_MultOperatorNodeType : Cast_AstNodeType<"Cast_MultOperatorNode", "multOperatorNode">{}
def Cast_DivOperatorNodeType : Cast_AstNodeType<"Cast_DivOperatorNode", "divOperatorNode">{}
def Cast_ModuloOperatorNodeType : Cast_AstNodeType<"Cast_ModuloOperatorNode", "moduloOperatorNode">{}
def Cast_CastExpressionNodeType : Cast_AstNodeType<"Cast_CastExpressionNode", "castExpressionNode">{}
def Cast_UnopExpressionNodeType : Cast_AstNodeType<"Cast_UnopExpressionNode", "unopExpressionNode">{}
def Cast_IncrOperatorNodeType : Cast_AstNodeType<"Cast_IncrOperatorNode", "incrOperatorNode">{}
def Cast_DecrOperatorNodeType : Cast_AstNodeType<"Cast_DecrOperatorNode", "decrOperatorNode">{}
def Cast_AddrofOperatorNodeType : Cast_AstNodeType<"Cast_AddrofOperatorNode", "addrofOperatorNode">{}
def Cast_DerefOperatorNodeType : Cast_AstNodeType<"Cast_DerefOperatorNode", "derefOperatorNode">{}
def Cast_PositiveOperatorNodeType : Cast_AstNodeType<"Cast_PositiveOperatorNode", "positiveOperatorNode">{}
def Cast_NegativeOperatorNodeType : Cast_AstNodeType<"Cast_NegativeOperatorNode", "negativeOperatorNode">{}
def Cast_NotOperatorNodeType : Cast_AstNodeType<"Cast_NotOperatorNode", "notOperatorNode">{}
def Cast_LnotOperatorNodeType : Cast_AstNodeType<"Cast_LnotOperatorNode", "lnotOperatorNode">{}
def Cast_SizeofExpressionNodeType : Cast_AstNodeType<"Cast_SizeofExpressionNode", "sizeofExpressionNode">{}
def Cast_PrimaryPostfixExpressionNodeType : Cast_AstNodeType<"Cast_PrimaryPostfixExpressionNode", "primaryPostfixExpressionNode">{}
def Cast_ArrayExpressionNodeType : Cast_AstNodeType<"Cast_ArrayExpressionNode", "arrayExpressionNode">{}
def Cast_FunCallExpressionNodeType : Cast_AstNodeType<"Cast_FunCallExpressionNode", "funCallExpressionNode">{}
def Cast_FieldExpressionNodeType : Cast_AstNodeType<"Cast_FieldExpressionNode", "fieldExpressionNode">{}
def Cast_PtrFieldExpressionNodeType : Cast_AstNodeType<"Cast_PtrFieldExpressionNode", "ptrFieldExpressionNode">{}
def Cast_PostincrExpressionNodeType : Cast_AstNodeType<"Cast_PostincrExpressionNode", "postincrExpressionNode">{}
def Cast_PostdecrExpressionNodeType : Cast_AstNodeType<"Cast_PostdecrExpressionNode", "postdecrExpressionNode">{}
def Cast_VarExpressionNodeType : Cast_AstNodeType<"Cast_VarExpressionNode", "varExpressionNode">{}
def Cast_IntExpressionNodeType : Cast_AstNodeType<"Cast_IntExpressionNode", "intExpressionNode">{}
def Cast_FloatExpressionNodeType : Cast_AstNodeType<"Cast_FloatExpressionNode", "floatExpressionNode">{}
def Cast_ParentExpressionNodeType : Cast_AstNodeType<"Cast_ParentExpressionNode", "parentExpressionNode">{}
def Cast_StructureExpressionNodeType : Cast_AstNodeType<"Cast_StructureExpressionNode", "structureExpressionNode">{}
def Cast_GenericExpressionNodeType : Cast_AstNodeType<"Cast_GenericExpressionNode", "genericExpressionNode">{}
def Cast_TypeGenericItemNodeType : Cast_AstNodeType<"Cast_TypeGenericItemNode", "typeGenericItemNode">{}
def Cast_DefaultGenericItemNodeType : Cast_AstNodeType<"Cast_DefaultGenericIteNode", "defaultGenericIteNode">{}


def Cast_ProgramPred : CPred<"$_self.isa<Cast_ProgramNodeType>()">;
def Cast_FunDefPred : CPred<"$_self.isa<Cast_FunDefNodeType>()">;
def Cast_FunProtoPred : CPred<"$_self.isa<Cast_FunProtoNodeType>()">;
def Cast_ArgumentPred : CPred<"$_self.isa<Cast_ArgumentNodeType>()">;
def Cast_VarDeclPred : CPred<"$_self.isa<Cast_VarDeclNodeType>()">;
def Cast_BaseVarDeclPred : CPred<"$_self.isa<Cast_BaseVarDeclNodeType>()">;
def Cast_UnionDefPred : CPred<"$_self.isa<Cast_UnionDefNodeType>()">;
def Cast_StructDefPred : CPred<"$_self.isa<Cast_StructDefNodeType>()">;
def Cast_EnumDefPred : CPred<"$_self.isa<Cast_EnumDefNodeType>()">;
def Cast_AliasDefPred : CPred<"$_self.isa<Cast_AliasDefNodeType>()">;
def Cast_EnumItemPred : CPred<"$_self.isa<Cast_EnumItemNodeType>()">;
def Cast_LabeledStatementPred : CPred<"$_self.isa<Cast_LabeledStatementNodeType>()">;
def Cast_CompoundStatementPred : CPred<"$_self.isa<Cast_CompoundStatementNodeType>()">;
def Cast_VarDeclStatementPred : CPred<"$_self.isa<Cast_VarDeclStatementNodeType>()">;
def Cast_ExpressionStatementPred : CPred<"$_self.isa<Cast_ExpressionStatementNodeType>()">;
def Cast_IfStatementPred : CPred<"$_self.isa<Cast_IfStatementNodeType>()">;
def Cast_SwitchStatementPred : CPred<"$_self.isa<Cast_SwitchStatementNodeType>()">;
def Cast_SwitchCaseItemPred : CPred<"$_self.isa<Cast_SwitchCaseItemNodeType>()">;
def Cast_SwitchDefaultItemPred : CPred<"$_self.isa<Cast_SwitchDefaultItemNodeType>()">;
def Cast_WhileStatementPred : CPred<"$_self.isa<Cast_WhileStatementNodeType>()">;
def Cast_DoWhileStatementPred : CPred<"$_self.isa<Cast_DoWhileStatementNodeType>()">;
def Cast_ForStatementPred : CPred<"$_self.isa<Cast_ForStatementNodeType>()">;
def Cast_GotoStatementPred : CPred<"$_self.isa<Cast_GotoStatementNodeType>()">;
def Cast_ContinueStatementPred : CPred<"$_self.isa<Cast_ContinueStatementNodeType>()">;
def Cast_BreakStatementPred : CPred<"$_self.isa<Cast_BreakStatementNodeType>()">;
def Cast_ReturnStatementPred : CPred<"$_self.isa<Cast_ReturnStatementNodeType>()">;
def Cast_TypePred : CPred<"$_self.isa<Cast_TypeNodeType>()">;
def Cast_ExternTypeModifierPred : CPred<"$_self.isa<Cast_ExternTypeModifierNodeType>()">;
def Cast_StaticTypeModifierPred : CPred<"$_self.isa<Cast_StaticTypeModifierNodeType>()">;
def Cast_RegisterTypeModifierPred : CPred<"$_self.isa<Cast_RegisterTypeModifierNodeType>()">;
def Cast_ConstTypeModifierPred : CPred<"$_self.isa<Cast_ConstTypeModifierNodeType>()">;
def Cast_VolatileTypeModifierPred : CPred<"$_self.isa<Cast_VolatileTypeModifierNodeType>()">;
def Cast_VoidTypePred : CPred<"$_self.isa<Cast_VoidTypeNodeType>()">;
def Cast_UnsignedLongLongTypePred : CPred<"$_self.isa<Cast_UnsignedLongLongTypeNodeType>()">;
def Cast_UnsignedLongTypePred : CPred<"$_self.isa<Cast_UnsignedLongTypeNodeType>()">;
def Cast_UnsignedShortTypePred : CPred<"$_self.isa<Cast_UnsignedShortTypeNodeType>()">;
def Cast_UnsignedCharTypePred : CPred<"$_self.isa<Cast_UnsignedCharTypeNodeType>()">;
def Cast_UnsignedIntTypePred : CPred<"$_self.isa<Cast_UnsignedIntTypeNodeType>()">;
def Cast_CharTypePred : CPred<"$_self.isa<Cast_CharTypeNodeType>()">;
def Cast_ShortTypePred : CPred<"$_self.isa<Cast_ShortTypeNodeType>()">;
def Cast_IntTypePred : CPred<"$_self.isa<Cast_IntTypeNodeType>()">;
def Cast_LongLongTypePred : CPred<"$_self.isa<Cast_LongLongTypeNodeType>()">;
def Cast_LongTypePred : CPred<"$_self.isa<Cast_LongTypeNodeType>()">;
def Cast_FloatTypePred : CPred<"$_self.isa<Cast_FloatTypeNodeType>()">;
def Cast_DoubleTypePred : CPred<"$_self.isa<Cast_DoubleTypeNodeType>()">;
def Cast_EnumTypePred : CPred<"$_self.isa<Cast_EnumTypeNodeType>()">;
def Cast_StructTypePred : CPred<"$_self.isa<Cast_StructTypeNodeType>()">;
def Cast_UnionTypePred : CPred<"$_self.isa<Cast_UnionTypeNodeType>()">;
def Cast_AliasTypePred : CPred<"$_self.isa<Cast_AliasTypeNodeType>()">;
def Cast_ExpressionPred : CPred<"$_self.isa<Cast_ExpressionNodeType>()">;
def Cast_AssignOpExpressionPred : CPred<"$_self.isa<Cast_AssignOpExpressionNodeType>()">;
def Cast_EqOpPred : CPred<"$_self.isa<Cast_EqOpNodeType>()">;
def Cast_StarEqOpPred : CPred<"$_self.isa<Cast_StarEqOpNodeType>()">;
def Cast_DivEqOpPred : CPred<"$_self.isa<Cast_DivEqOpNodeType>()">;
def Cast_ModuloEqOpPred : CPred<"$_self.isa<Cast_ModuloEqOpNodeType>()">;
def Cast_PlusEqOpPred : CPred<"$_self.isa<Cast_PlusEqOpNodeType>()">;
def Cast_MinusEqOpPred : CPred<"$_self.isa<Cast_MinusEqOpNodeType>()">;
def Cast_LeftShiftEqOpPred : CPred<"$_self.isa<Cast_LeftShiftEqOpNodeType>()">;
def Cast_RightShiftEqOpPred : CPred<"$_self.isa<Cast_RightShiftEqOpNodeType>()">;
def Cast_AndEqOpPred : CPred<"$_self.isa<Cast_AndEqOpNodeType>()">;
def Cast_XorEqOpPred : CPred<"$_self.isa<Cast_XorEqOpNodeType>()">;
def Cast_OrEqOpPred : CPred<"$_self.isa<Cast_OrEqOpNodeType>()">;
def Cast_ConditionalOpExpressionPred : CPred<"$_self.isa<Cast_ConditionalOpExpressionNodeType>()">;
def Cast_LogicalOrExpressionPred : CPred<"$_self.isa<Cast_LogicalOrExpressionNodeType>()">;
def Cast_LogicalAndExpressionPred : CPred<"$_self.isa<Cast_LogicalAndExpressionNodeType>()">;
def Cast_OrExpressionPred : CPred<"$_self.isa<Cast_OrExpressionNodeType>()">;
def Cast_XorExpressionPred : CPred<"$_self.isa<Cast_XorExpressionNodeType>()">;
def Cast_AndExpressionPred : CPred<"$_self.isa<Cast_AndExpressionNodeType>()">;
def Cast_EqualOpExpressionPred : CPred<"$_self.isa<Cast_EqualOpExpressionNodeType>()">;
def Cast_EqualOperatorPred : CPred<"$_self.isa<Cast_EqualOperatorNodeType>()">;
def Cast_NotEqualOperatorPred : CPred<"$_self.isa<Cast_NotEqualOperatorNodeType>()">;
def Cast_RelOpExpressionPred : CPred<"$_self.isa<Cast_RelOpExpressionNodeType>()">;
def Cast_GeOperatorPred : CPred<"$_self.isa<Cast_GeOperatorNodeType>()">;
def Cast_GtOperatorPred : CPred<"$_self.isa<Cast_GtOperatorNodeType>()">;
def Cast_LeOperatorPred : CPred<"$_self.isa<Cast_LeOperatorNodeType>()">;
def Cast_LtOperatorPred : CPred<"$_self.isa<Cast_LtOperatorNodeType>()">;
def Cast_ShiftOpExpressionPred : CPred<"$_self.isa<Cast_ShiftOpExpressionNodeType>()">;
def Cast_LshiftOperatorPred : CPred<"$_self.isa<Cast_LshiftOperatorNodeType>()">;
def Cast_RshiftOperatorPred : CPred<"$_self.isa<Cast_RshiftOperatorNodeType>()">;
def Cast_AddOpExpressionPred : CPred<"$_self.isa<Cast_AddOpExpressionNodeType>()">;
def Cast_PlusOperatorPred : CPred<"$_self.isa<Cast_PlusOperatorNodeType>()">;
def Cast_MinusOperatorPred : CPred<"$_self.isa<Cast_MinusOperatorNodeType>()">;
def Cast_MulOpExpressionPred : CPred<"$_self.isa<Cast_MulOpExpressionNodeType>()">;
def Cast_MultOperatorPred : CPred<"$_self.isa<Cast_MultOperatorNodeType>()">;
def Cast_DivOperatorPred : CPred<"$_self.isa<Cast_DivOperatorNodeType>()">;
def Cast_ModuloOperatorPred : CPred<"$_self.isa<Cast_ModuloOperatorNodeType>()">;
def Cast_CastExpressionPred : CPred<"$_self.isa<Cast_CastExpressionNodeType>()">;
def Cast_UnopExpressionPred : CPred<"$_self.isa<Cast_UnopExpressionNodeType>()">;
def Cast_IncrOperatorPred : CPred<"$_self.isa<Cast_IncrOperatorNodeType>()">;
def Cast_DecrOperatorPred : CPred<"$_self.isa<Cast_DecrOperatorNodeType>()">;
def Cast_AddrofOperatorPred : CPred<"$_self.isa<Cast_AddrofOperatorNodeType>()">;
def Cast_DerefOperatorPred : CPred<"$_self.isa<Cast_DerefOperatorNodeType>()">;
def Cast_PositiveOperatorPred : CPred<"$_self.isa<Cast_PositiveOperatorNodeType>()">;
def Cast_NegativeOperatorPred : CPred<"$_self.isa<Cast_NegativeOperatorNodeType>()">;
def Cast_NotOperatorPred : CPred<"$_self.isa<Cast_NotOperatorNodeType>()">;
def Cast_LnotOperatorPred : CPred<"$_self.isa<Cast_LnotOperatorNodeType>()">;
def Cast_SizeofExpressionPred : CPred<"$_self.isa<Cast_SizeofExpressionNodeType>()">;
def Cast_PrimaryPostfixExpressionPred : CPred<"$_self.isa<Cast_PrimaryPostfixExpressionNodeType>()">;
def Cast_ArrayExpressionPred : CPred<"$_self.isa<Cast_ArrayExpressionNodeType>()">;
def Cast_FunCallExpressionPred : CPred<"$_self.isa<Cast_FunCallExpressionNodeType>()">;
def Cast_FieldExpressionPred : CPred<"$_self.isa<Cast_FieldExpressionNodeType>()">;
def Cast_PtrFieldExpressionPred : CPred<"$_self.isa<Cast_PtrFieldExpressionNodeType>()">;
def Cast_PostincrExpressionPred : CPred<"$_self.isa<Cast_PostincrExpressionNodeType>()">;
def Cast_PostdecrExpressionPred : CPred<"$_self.isa<Cast_PostdecrExpressionNodeType>()">;
def Cast_VarExpressionPred : CPred<"$_self.isa<Cast_VarExpressionNodeType>()">;
def Cast_IntExpressionPred : CPred<"$_self.isa<Cast_IntExpressionNodeType>()">;
def Cast_FloatExpressionPred : CPred<"$_self.isa<Cast_FloatExpressionNodeType>()">;
def Cast_ParentExpressionPred : CPred<"$_self.isa<Cast_ParentExpressionNodeType>()">;
def Cast_StructureExpressionPred : CPred<"$_self.isa<Cast_StructureExpressionNodeType>()">;
def Cast_GenericExpressionPred : CPred<"$_self.isa<Cast_GenericExpressionNodeType>()">;
def Cast_TypeGenericItemPred : CPred<"$_self.isa<Cast_TypeGenericItemNodeType>()">;
def Cast_DefaultGenericItemPred : CPred<"$_self.isa<Cast_DefaultGenericItemNodeType>()">;

def Cast_TypeDefPred : Or<[
  Cast_UnionDefPred,
  Cast_StructDefPred,
  Cast_EnumDefPred,
  Cast_AliasDefPred
]>;

def Cast_StatementPred : Or<[
  Cast_LabeledStatementPred,
  Cast_CompoundStatementPred,
  Cast_VarDeclStatementPred,
  Cast_ExpressionStatementPred,
  Cast_IfStatementPred,
  Cast_SwitchStatementPred,
  Cast_WhileStatementPred,
  Cast_DoWhileStatementPred,
  Cast_ForStatementPred,
  Cast_GotoStatementPred,
  Cast_ContinueStatementPred,
  Cast_BreakStatementPred,
  Cast_ReturnStatementPred
]>;

def Cast_SwitchItemPred : Or<[
  Cast_SwitchCaseItemPred,
  Cast_SwitchDefaultItemPred
]>;

def Cast_TypeModifierPred : Or<[
  Cast_ExternTypeModifierPred,
  Cast_StaticTypeModifierPred,
  Cast_RegisterTypeModifierPred,
  Cast_ConstTypeModifierPred,
  Cast_VolatileTypeModifierPred
]>;

def Cast_BaseTypePred : Or<[
  Cast_VoidTypePred,
  Cast_UnsignedLongLongTypePred,
  Cast_UnsignedLongTypePred,
  Cast_UnsignedShortTypePred,
  Cast_UnsignedCharTypePred,
  Cast_UnsignedIntTypePred,
  Cast_CharTypePred,
  Cast_ShortTypePred,
  Cast_IntTypePred,
  Cast_LongLongTypePred,
  Cast_LongTypePred,
  Cast_FloatTypePred,
  Cast_DoubleTypePred,
  Cast_EnumTypePred,
  Cast_StructTypePred,
  Cast_UnionTypePred,
  Cast_AliasTypePred
]>;

def Cast_AssignOperatorPred : Or<[
  Cast_EqOpPred,
  Cast_StarEqOpPred,
  Cast_DivEqOpPred,
  Cast_ModuloEqOpPred,
  Cast_PlusEqOpPred,
  Cast_MinusEqOpPred,
  Cast_LeftShiftEqOpPred,
  Cast_RightShiftEqOpPred,
  Cast_AndEqOpPred,
  Cast_XorEqOpPred,
  Cast_OrEqOpPred
  ]>;

def Cast_EqualityOperatorPred : Or<[
  Cast_EqualOperatorPred,
  Cast_NotEqualOperatorPred
]>;

def Cast_RelationalOperatorPred : Or<[
  Cast_GeOperatorPred,
  Cast_GtOperatorPred,
  Cast_LeOperatorPred,
  Cast_LtOperatorPred
]>;

def Cast_ShiftOperatorPred : Or<[
  Cast_LshiftOperatorPred,
  Cast_RshiftOperatorPred
]>;

def Cast_AdditiveOperatorPred : Or<[
  Cast_PlusOperatorPred,
  Cast_MinusOperatorPred
]>;

def Cast_MultiplicativeOperatorPred : Or<[
  Cast_MultOperatorPred,
  Cast_DivOperatorPred,
  Cast_ModuloOperatorPred
]>;

def Cast_UnaryOperatorPred : Or<[
  Cast_IncrOperatorPred,
  Cast_DecrOperatorPred,
  Cast_AddrofOperatorPred,
  Cast_DerefOperatorPred,
  Cast_PositiveOperatorPred,
  Cast_NegativeOperatorPred,
  Cast_NotOperatorPred,
  Cast_LnotOperatorPred
]>;

def Cast_PostfixExpressionPred : Or<[
  Cast_PrimaryPostfixExpressionPred,
  Cast_ArrayExpressionPred,
  Cast_FunCallExpressionPred,
  Cast_FieldExpressionPred,
  Cast_PtrFieldExpressionPred,
  Cast_PostincrExpressionPred,
  Cast_PostdecrExpressionPred
]>;

def Cast_PrimaryExpression : Or<[
  Cast_VarExpressionPred,
  Cast_IntExpressionPred,
  Cast_FloatExpressionPred,
  Cast_ParentExpressionPred,
  Cast_StructureExpressionPred,
  Cast_GenericExpressionPred
]>;

def Cast_GenericItemPred : Or<[
  Cast_TypeGenericItemPred,
  Cast_DefaultGenericItemPred
]>;

def Cast_ProgramItemPred : Or<[
  Cast_FunDefPred,
  Cast_VarDeclPred,
  Cast_TypeDefPred
]>;


def Cast_ConditionalExpressionPred : Or<[
  Cast_LogicalOrExpressionPred,
  Cast_ConditionalOpExpressionPred
]>;

def Cast_AssignmentExpressionPred : Or<[
  Cast_ConditionalExpressionPred,
  Cast_AssignOpExpressionPred
]>;

def Cast_MultiplicativeExpressionPred : Or<[
  Cast_CastExpressionPred,
  Cast_MulOpExpressionPred
]>;

def Cast_AdditiveExpressionPred : Or<[
  Cast_MultiplicativeExpressionPred,
  Cast_AddOpExpressionPred
]>;

def Cast_ShiftExpressionPred : Or<[
  Cast_AdditiveExpressionPred,
  Cast_ShiftOpExpressionPred
]>;

def Cast_RelationalExpressionPred :  Or<[
  Cast_ShiftExpressionPred,
  Cast_RelOpExpressionPred
]>;

def Cast_EqualityExpressionPred : Or<[
  Cast_RelationalExpressionPred,
  Cast_EqualOpExpressionPred
]>;

def Cast_UnaryExpressionPred : Or<[
  Cast_PostfixExpressionPred,
  Cast_SizeofExpressionPred,
  Cast_UnopExpressionPred
]>;

#endif
