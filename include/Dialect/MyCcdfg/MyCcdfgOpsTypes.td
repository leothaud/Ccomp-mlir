//===--------------------------MyCcdfgOpsTypes.td--------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_TYPES_TD__
#define MYCCDFG_TYPES_TD__

include "Dialect/MyCcdfg/MyCcdfgDialect.td"
include "mlir/IR/AttrTypeBase.td"

class MyCcdfg_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<MyCcdfg_Dialect, name, traits>
{
    let mnemonic = typeMnemonic;
}


def MyCcdfg_ProgramType : MyCcdfg_Type<"MyCcdfg_Program", "program">{}
def MyCcdfg_ScopeType : MyCcdfg_Type<"MyCcdfg_Scope", "scope">{}

def MyCcdfg_FunDeclType : MyCcdfg_Type<"MyCcdfg_funDecl", "funDecl">{}

def MyCcdfg_FunDefType : MyCcdfg_Type<"MyCcdfg_funDef", "funDef">{}

def MyCcdfg_VarDeclType : MyCcdfg_Type<"MyCcdfg_VarDecl", "varDecl">{}

def MyCcdfg_TypeDefType : MyCcdfg_Type<"MyCcdfg_TypeDef", "typeDef">{}
def MyCcdfg_UnionDefType : MyCcdfg_Type<"MyCcdfg_UnionDef", "unionDef">{}
def MyCcdfg_StructDefType : MyCcdfg_Type<"MyCcdfg_StructDef", "structDef">{}
def MyCcdfg_EnumDefType : MyCcdfg_Type<"MyCcdfg_EnumDef", "enumDef">{}



def MyCcdfg_TypeType : MyCcdfg_Type<"MyCcdfg_Type", "type">{}
def MyCcdfg_StaticTypeType : MyCcdfg_Type<"MyCcdfg_StaticType", "staticType">{}
def MyCcdfg_ConstTypeType : MyCcdfg_Type<"MyCcdfg_ConstType", "constType">{}
def MyCcdfg_ExternTypeType : MyCcdfg_Type<"MyCcdfg_ExternType", "externType">{}
def MyCcdfg_VolatileTypeType : MyCcdfg_Type<"MyCcdfg_VolatileType", "volatileType">{}
def MyCcdfg_PointerTypeType : MyCcdfg_Type<"MyCcdfg_PointerType", "pointerType">{}
def MyCcdfg_ArrayTypeType : MyCcdfg_Type<"MyCcdfg_ArrayType", "arrayType">{}
def MyCcdfg_VoidTypeType : MyCcdfg_Type<"MyCcdfg_VoidType", "voidType">{}
def MyCcdfg_IntTypeType : MyCcdfg_Type<"MyCcdfg_IntType", "intType">{}
def MyCcdfg_FloatTypeType : MyCcdfg_Type<"MyCcdfg_FloatType", "floatType">{}
def MyCcdfg_DoubleTypeType : MyCcdfg_Type<"MyCcdfg_DoubleType", "doubleType">{}
def MyCcdfg_EnumTypeType : MyCcdfg_Type<"MyCcdfg_EnumType", "enumType">{}
def MyCcdfg_UnionTypeType : MyCcdfg_Type<"MyCcdfg_UnionType", "unionType">{}
def MyCcdfg_StructTypeType : MyCcdfg_Type<"MyCcdfg_StructType", "structType">{}
 
def MyCcdfg_ExpressionType : MyCcdfg_Type<"MyCcdfg_Expression", "expression">{}
def MyCcdfg_AssignExpressionType : MyCcdfg_Type<"MyCcdfg_AssignExpression", "assignExpression">{}
def MyCcdfg_MuxExpressionType : MyCcdfg_Type<"MyCcdfg_MuxExpression", "muxExpression">{}
def MyCcdfg_LandExpressionType : MyCcdfg_Type<"MyCcdfg_LandExpression", "landExpression">{}
def MyCcdfg_LorExpressionType : MyCcdfg_Type<"MyCcdfg_LorExpression", "lorExpression">{}
def MyCcdfg_AndExpressionType : MyCcdfg_Type<"MyCcdfg_AndExpression", "andExpression">{}
def MyCcdfg_OrExpressionType : MyCcdfg_Type<"MyCcdfg_OrExpression", "orExpression">{}
def MyCcdfg_XorExpressionType : MyCcdfg_Type<"MyCcdfg_XorExpression", "xorExpression">{}
def MyCcdfg_EqExpressionType : MyCcdfg_Type<"MyCcdfg_EqExpression", "eqExpression">{}
def MyCcdfg_NeqExpressionType : MyCcdfg_Type<"MyCcdfg_NeqExpression", "neqExpression">{}
def MyCcdfg_GeExpressionType : MyCcdfg_Type<"MyCcdfg_GeExpression", "geExpression">{}
def MyCcdfg_GtExpressionType : MyCcdfg_Type<"MyCcdfg_GtExpression", "gtExpression">{}
def MyCcdfg_LeExpressionType : MyCcdfg_Type<"MyCcdfg_LeExpression", "leExpression">{}
def MyCcdfg_LtExpressionType : MyCcdfg_Type<"MyCcdfg_LtExpression", "ltExpression">{}
def MyCcdfg_LshiftExpressionType : MyCcdfg_Type<"MyCcdfg_LshiftExpression", "lshiftExpression">{}
def MyCcdfg_RshiftExpressionType : MyCcdfg_Type<"MyCcdfg_RshiftExpression", "rshiftExpression">{}
def MyCcdfg_AddExpressionType : MyCcdfg_Type<"MyCcdfg_AddExpression", "addExpression">{}
def MyCcdfg_SubExpressionType : MyCcdfg_Type<"MyCcdfg_SubExpression", "subExpression">{}
def MyCcdfg_CastExpressionType : MyCcdfg_Type<"MyCcdfg_CastExpression", "castExpression">{}
def MyCcdfg_MultExpressionType : MyCcdfg_Type<"MyCcdfg_MultExpression", "multExpression">{}
def MyCcdfg_DivExpressionType : MyCcdfg_Type<"MyCcdfg_DivExpression", "divExpression">{}
def MyCcdfg_ModuloExpressionType : MyCcdfg_Type<"MyCcdfg_ModuloExpression", "moduloExpression">{}
def MyCcdfg_SizeofExprExpressionType : MyCcdfg_Type<"MyCcdfg_SizeofExprExpression", "sizeofExpr">{}
def MyCcdfg_SizeofTypeExpressionType : MyCcdfg_Type<"MyCcdfg_SizeofTypeExpression", "sizeofType">{}
def MyCcdfg_DerefExpressionType : MyCcdfg_Type<"MyCcdfg_DerefExpression", "derefExpression">{}
def MyCcdfg_AddrofExpressionType : MyCcdfg_Type<"MyCcdfg_AddrofExpression", "addrofExpression">{}
def MyCcdfg_NegExpressionType : MyCcdfg_Type<"MyCcdfg_NegExpression", "negExpression">{}
def MyCcdfg_NotExpressionType : MyCcdfg_Type<"MyCcdfg_NotExpression", "notExpression">{}
def MyCcdfg_LnotExpressionType : MyCcdfg_Type<"MyCcdfg_LnotExpression", "lnotExpression">{}
def MyCcdfg_FieldExpressionType : MyCcdfg_Type<"MyCcdfg_FieldExpression", "fieldExpression">{}
def MyCcdfg_PtrFieldExpressionType : MyCcdfg_Type<"MyCcdfg_PtrFieldExpression", "ptrFieldExpression">{}
def MyCcdfg_FunCallExpressionType : MyCcdfg_Type<"MyCcdfg_FunCallExpression", "funCallExpression">{}
def MyCcdfg_ArrayExpressionType : MyCcdfg_Type<"MyCcdfg_ArrayExpression", "arrayExpression">{}

def MyCcdfg_ConstExpressionType : MyCcdfg_Type<"MyCcdfg_ConstExpression", "constExpression">{}
def MyCcdfg_IntConstExpressionType : MyCcdfg_Type<"MyCcdfg_IntConstExpression", "intConstExpression">{}
def MyCcdfg_FloatConstExpressionType : MyCcdfg_Type<"MyCcdfg_FloatConstExpression", "floatConstExpression">{}
def MyCcdfg_StringConstExpressionType : MyCcdfg_Type<"MyCcdfg_StringConstExpression", "stringConstExpression">{}
def MyCcdfg_StructureConstExpressionType : MyCcdfg_Type<"MyCcdfg_StructureConstExpression", "structureConstExpression">{}

def MyCcdfg_GenericExpressionType : MyCcdfg_Type<"MyCcdfg_GenericExpression", "genericExpression">{}
def MyCcdfg_CaseGenericExpressionType : MyCcdfg_Type<"MyCcdfg_CaseGenericExpression", "caseGenericExpression">{}
def MyCcdfg_DefaultGenericExpressionType : MyCcdfg_Type<"MyCcdfg_DefaultGenericExpression", "defaultGenericExpression">{}

def MyCcdfg_LvalueType : MyCcdfg_Type<"MyCcdfg_Lvalue", "lvalue">{}
def MyCcdfg_LvalueVarType : MyCcdfg_Type<"MyCcdfg_LvalueVar", "lvalueVar">{}
def MyCcdfg_LvalueArrayType : MyCcdfg_Type<"MyCcdfg_LvalueArray", "lvalueArray">{}
def MyCcdfg_LvalueDerefType : MyCcdfg_Type<"MyCcdfg_LvalueDeref", "lvalueDeref">{}
def MyCcdfg_LvalueFieldType : MyCcdfg_Type<"MyCcdfg_LvalueField", "lvalueField">{}
def MyCcdfg_LvaluePtrFieldType : MyCcdfg_Type<"MyCcdfg_LvaluePtrField", "lvaluePtrField">{}

def MyCcdfg_BasicBlockType : MyCcdfg_Type<"MyCcdfg_BasicBlock", "basicBlock">{}

def MyCcdfg_TransitionType : MyCcdfg_Type<"MyCcdfg_Transition", "transition">{}
def MyCcdfg_UnconditionalTransitionType : MyCcdfg_Type<"MyCcdfg_UnconditionalTransition", "unconditionalTransition">{}
def MyCcdfg_ConditionalTransitionType : MyCcdfg_Type<"MyCcdfg_ConditionalTransition", "conditionalTransition">{}


#endif // MYCCDFG_TYPES_TD__