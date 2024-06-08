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
include "mlir/IR/Constraints.td"
include "mlir/IR/BuiltinAttributes.td"

class MyCcdfg_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<MyCcdfg_Dialect, name, traits>
{
    let mnemonic = typeMnemonic;
}

def StringArrayAttr : TypedArrayAttrBase<Builtin_StringAttr,
                                      "String array attribute"> {
  let constBuilderCall = "$_builder.getStringArrayAttr($0)";
}



def MyCcdfg_ProgramType : MyCcdfg_Type<"MyCcdfg_Program", "program">{}
def MyCcdfg_ScopeType : MyCcdfg_Type<"MyCcdfg_Scope", "scope">{}

def MyCcdfg_FunDeclType : MyCcdfg_Type<"MyCcdfg_FunDecl", "funDecl">{}

def MyCcdfg_FunDefType : MyCcdfg_Type<"MyCcdfg_FunDef", "funDef">{}

def MyCcdfg_TypeDefType : MyCcdfg_Type<"MyCcdfg_TypeDef", "typeDef">{}
def MyCcdfg_UnionDefType : MyCcdfg_Type<"MyCcdfg_UnionDef", "unionDef">{}
def MyCcdfg_StructDefType : MyCcdfg_Type<"MyCcdfg_StructDef", "structDef">{}
def MyCcdfg_EnumDefType : MyCcdfg_Type<"MyCcdfg_EnumDef", "enumDef">{}

def MyCcdfg_TypeType : MyCcdfg_Type<"MyCcdfg_Type", "type">{}
def MyCcdfg_ModifiedTypeType : MyCcdfg_Type<"MyCcdfg_ModifiedType", "modifiedType">{}

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
def MyCcdfg_SizeofExprExpressionType : MyCcdfg_Type<"MyCcdfg_SizeofExprExpression", "sizeofExprExpression">{}
def MyCcdfg_SizeofTypeExpressionType : MyCcdfg_Type<"MyCcdfg_SizeofTypeExpression", "sizeofTypeExpression">{}
def MyCcdfg_DerefExpressionType : MyCcdfg_Type<"MyCcdfg_DerefExpression", "derefExpression">{}
def MyCcdfg_AddrofExpressionType : MyCcdfg_Type<"MyCcdfg_AddrofExpression", "addrofExpression">{}
def MyCcdfg_NegExpressionType : MyCcdfg_Type<"MyCcdfg_NegExpression", "negExpression">{}
def MyCcdfg_NotExpressionType : MyCcdfg_Type<"MyCcdfg_NotExpression", "notExpression">{}
def MyCcdfg_LnotExpressionType : MyCcdfg_Type<"MyCcdfg_LnotExpression", "lnotExpression">{}
def MyCcdfg_FieldExpressionType : MyCcdfg_Type<"MyCcdfg_FieldExpression", "fieldExpression">{}
def MyCcdfg_PtrFieldExpressionType : MyCcdfg_Type<"MyCcdfg_PtrFieldExpression", "ptrFieldExpression">{}
def MyCcdfg_FunCallExpressionType : MyCcdfg_Type<"MyCcdfg_FunCallExpression", "funCallExpression">{}
def MyCcdfg_ArrayExpressionType : MyCcdfg_Type<"MyCcdfg_ArrayExpression", "arrayExpression">{}

def MyCcdfg_VarExpressionType : MyCcdfg_Type<"MyCcdfg_VarExpression", "varExpression">{}

def MyCcdfg_ConstExpressionType : MyCcdfg_Type<"MyCcdfg_ConstExpression", "constExpression">{}
def MyCcdfg_IntConstExpressionType : MyCcdfg_Type<"MyCcdfg_IntConstExpression", "intConstExpression">{}
def MyCcdfg_FloatConstExpressionType : MyCcdfg_Type<"MyCcdfg_FloatConstExpression", "floatConstExpression">{}
def MyCcdfg_StringConstExpressionType : MyCcdfg_Type<"MyCcdfg_StringConstExpression", "stringConstExpression">{}
def MyCcdfg_StructureConstExpressionType : MyCcdfg_Type<"MyCcdfg_StructureConstExpression", "structureConstExpression">{}

def MyCcdfg_LvalueType : MyCcdfg_Type<"MyCcdfg_Lvalue", "lvalue">{}
def MyCcdfg_LvalueVarType : MyCcdfg_Type<"MyCcdfg_LvalueVar", "lvalueVar">{}
def MyCcdfg_LvalueArrayType : MyCcdfg_Type<"MyCcdfg_LvalueArray", "lvalueArray">{}
def MyCcdfg_LvalueDerefType : MyCcdfg_Type<"MyCcdfg_LvalueDeref", "lvalueDeref">{}
def MyCcdfg_LvalueFieldType : MyCcdfg_Type<"MyCcdfg_LvalueField", "lvalueField">{}
def MyCcdfg_LvaluePtrFieldType : MyCcdfg_Type<"MyCcdfg_LvaluePtrField", "lvaluePtrField">{}

def MyCcdfg_BasicBlockType : MyCcdfg_Type<"MyCcdfg_BasicBlock", "basicBlock">{}

def MyCcdfg_InstrType : MyCcdfg_Type<"MyCcdfg_Instr", "instr">{}
def MyCcdfg_FunCallInstrType : MyCcdfg_Type<"MyCcdfg_FunCallInstr", "funCallInstr">{}
def MyCcdfg_AffectInstrType : MyCcdfg_Type<"MyCcdfg_AffectInstr", "affectInstr">{}

def MyCcdfg_TransitionType : MyCcdfg_Type<"MyCcdfg_Transition", "transition">{}
def MyCcdfg_InputTransitionType : MyCcdfg_Type<"MyCcdfg_InputTransition", "inputTransition">{}
def MyCcdfg_OutputTransitionType : MyCcdfg_Type<"MyCcdfg_OutputTransition", "outputTransition">{}
def MyCcdfg_UnconditionalTransitionType : MyCcdfg_Type<"MyCcdfg_UnconditionalTransition", "unconditionalTransition">{}
def MyCcdfg_ConditionalTransitionType : MyCcdfg_Type<"MyCcdfg_ConditionalTransition", "conditionalTransition">{}

def MyCcdfg_ProgramPred : CPred<"mlir::isa<MyCcdfg_ProgramType>($_self)">;
def MyCcdfg_ScopePred : CPred<"mlir::isa<MyCcdfg_ScopeType>($_self)">;

def MyCcdfg_FunDeclPred : CPred<"mlir::isa<MyCcdfg_FunDeclType>($_self)">;
def MyCcdfg_FunDefPred : CPred<"mlir::isa<MyCcdfg_FunDefType>($_self)">;

def MyCcdfg_UnionDefPred : CPred<"mlir::isa<MyCcdfg_UnionDefType>($_self)">;
def MyCcdfg_StructDefPred : CPred<"mlir::isa<MyCcdfg_StructDefType>($_self)">;
def MyCcdfg_EnumDefPred : CPred<"mlir::isa<MyCcdfg_EnumDefType>($_self)">;
def MyCcdfg_TypeDefPred : Or<[
    CPred<"mlir::isa<MyCcdfg_TypeDefType>($_self)">,
    MyCcdfg_UnionDefPred,
    MyCcdfg_StructDefPred,
    MyCcdfg_EnumDefPred
]>;




def MyCcdfg_ModifiedTypePred : CPred<"mlir::isa<MyCcdfg_ModifiedTypeType>($_self)">;
def MyCcdfg_PointerTypePred : CPred<"mlir::isa<MyCcdfg_PointerTypeType>($_self)">;
def MyCcdfg_ArrayTypePred : CPred<"mlir::isa<MyCcdfg_ArrayTypeType>($_self)">;
def MyCcdfg_VoidTypePred : CPred<"mlir::isa<MyCcdfg_VoidTypeType>($_self)">;
def MyCcdfg_IntTypePred : CPred<"mlir::isa<MyCcdfg_IntTypeType>($_self)">;
def MyCcdfg_FloatTypePred : CPred<"mlir::isa<MyCcdfg_FloatTypeType>($_self)">;
def MyCcdfg_DoubleTypePred : CPred<"mlir::isa<MyCcdfg_DoubleTypeType>($_self)">;
def MyCcdfg_EnumTypePred : CPred<"mlir::isa<MyCcdfg_EnumTypeType>($_self)">;
def MyCcdfg_UnionTypePred : CPred<"mlir::isa<MyCcdfg_UnionTypeType>($_self)">;
def MyCcdfg_StructTypePred : CPred<"mlir::isa<MyCcdfg_StructTypeType>($_self)">;


def MyCcdfg_TypePred : Or<[
    CPred<"mlir::isa<MyCcdfg_TypeType>($_self)">,
    MyCcdfg_ModifiedTypePred,
    MyCcdfg_PointerTypePred,
    MyCcdfg_ArrayTypePred,
    MyCcdfg_VoidTypePred,
    MyCcdfg_IntTypePred,
    MyCcdfg_FloatTypePred,
    MyCcdfg_DoubleTypePred,
    MyCcdfg_EnumTypePred,
    MyCcdfg_UnionTypePred,
    MyCcdfg_StructTypePred
]>;


def MyCcdfg_MuxExpressionPred : CPred<"mlir::isa<MyCcdfg_MuxExpressionType>($_self)">;
def MyCcdfg_LandExpressionPred : CPred<"mlir::isa<MyCcdfg_LandExpressionType>($_self)">;
def MyCcdfg_LorExpressionPred : CPred<"mlir::isa<MyCcdfg_LorExpressionType>($_self)">;
def MyCcdfg_AndExpressionPred : CPred<"mlir::isa<MyCcdfg_AndExpressionType>($_self)">;
def MyCcdfg_OrExpressionPred : CPred<"mlir::isa<MyCcdfg_OrExpressionType>($_self)">;
def MyCcdfg_XorExpressionPred : CPred<"mlir::isa<MyCcdfg_XorExpressionType>($_self)">;
def MyCcdfg_EqExpressionPred : CPred<"mlir::isa<MyCcdfg_EqExpressionType>($_self)">;
def MyCcdfg_NeqExpressionPred : CPred<"mlir::isa<MyCcdfg_NeqExpressionType>($_self)">;
def MyCcdfg_GeExpressionPred : CPred<"mlir::isa<MyCcdfg_GeExpressionType>($_self)">;
def MyCcdfg_GtExpressionPred : CPred<"mlir::isa<MyCcdfg_GtExpressionType>($_self)">;
def MyCcdfg_LeExpressionPred : CPred<"mlir::isa<MyCcdfg_LeExpressionType>($_self)">;
def MyCcdfg_LtExpressionPred : CPred<"mlir::isa<MyCcdfg_LtExpressionType>($_self)">;
def MyCcdfg_LshiftExpressionPred : CPred<"mlir::isa<MyCcdfg_LshiftExpressionType>($_self)">;
def MyCcdfg_RshiftExpressionPred : CPred<"mlir::isa<MyCcdfg_RshiftExpressionType>($_self)">;
def MyCcdfg_AddExpressionPred : CPred<"mlir::isa<MyCcdfg_AddExpressionType>($_self)">;
def MyCcdfg_SubExpressionPred : CPred<"mlir::isa<MyCcdfg_SubExpressionType>($_self)">;
def MyCcdfg_CastExpressionPred : CPred<"mlir::isa<MyCcdfg_CastExpressionType>($_self)">;
def MyCcdfg_MultExpressionPred : CPred<"mlir::isa<MyCcdfg_MultExpressionType>($_self)">;
def MyCcdfg_DivExpressionPred : CPred<"mlir::isa<MyCcdfg_DivExpressionType>($_self)">;
def MyCcdfg_ModuloExpressionPred : CPred<"mlir::isa<MyCcdfg_ModuloExpressionType>($_self)">;
def MyCcdfg_SizeofExprExpressionPred : CPred<"mlir::isa<MyCcdfg_SizeofExprExpressionType>($_self)">;
def MyCcdfg_SizeofTypeExpressionPred : CPred<"mlir::isa<MyCcdfg_SizeofTypeExpressionType>($_self)">;
def MyCcdfg_DerefExpressionPred : CPred<"mlir::isa<MyCcdfg_DerefExpressionType>($_self)">;
def MyCcdfg_AddrofExpressionPred : CPred<"mlir::isa<MyCcdfg_AddrofExpressionType>($_self)">;
def MyCcdfg_NegExpressionPred : CPred<"mlir::isa<MyCcdfg_NegExpressionType>($_self)">;
def MyCcdfg_NotExpressionPred : CPred<"mlir::isa<MyCcdfg_NotExpressionType>($_self)">;
def MyCcdfg_LnotExpressionPred : CPred<"mlir::isa<MyCcdfg_LnotExpressionType>($_self)">;
def MyCcdfg_FieldExpressionPred : CPred<"mlir::isa<MyCcdfg_FieldExpressionType>($_self)">;
def MyCcdfg_PtrFieldExpressionPred : CPred<"mlir::isa<MyCcdfg_PtrFieldExpressionType>($_self)">;
def MyCcdfg_FunCallExpressionPred : CPred<"mlir::isa<MyCcdfg_FunCallExpressionType>($_self)">;
def MyCcdfg_ArrayExpressionPred : CPred<"mlir::isa<MyCcdfg_ArrayExpressionType>($_self)">;

def MyCcdfg_VarExpressionPred : CPred<"mlir::isa<MyCcdfg_VarExpressionType>($_self)">;

def MyCcdfg_IntConstExpressionPred : CPred<"mlir::isa<MyCcdfg_IntConstExpressionType>($_self)">;
def MyCcdfg_FloatConstExpressionPred : CPred<"mlir::isa<MyCcdfg_FloatConstExpressionType>($_self)">;
def MyCcdfg_StringConstExpressionPred : CPred<"mlir::isa<MyCcdfg_StringConstExpressionType>($_self)">;
def MyCcdfg_StructureConstExpressionPred : CPred<"mlir::isa<MyCcdfg_StructureConstExpressionType>($_self)">;


def MyCcdfg_ConstExpressionPred : Or<[
    CPred<"mlir::isa<MyCcdfg_ConstExpressionType>($_self)">,
    MyCcdfg_IntConstExpressionPred,
    MyCcdfg_FloatConstExpressionPred,
    MyCcdfg_StringConstExpressionPred,
    MyCcdfg_StructureConstExpressionPred
]>;

def MyCcdfg_ExpressionPred : Or<[
    CPred<"mlir::isa<MyCcdfg_ExpressionType>($_self)">,
    MyCcdfg_MuxExpressionPred,
    MyCcdfg_LandExpressionPred,
    MyCcdfg_LorExpressionPred,
    MyCcdfg_AndExpressionPred,
    MyCcdfg_OrExpressionPred,
    MyCcdfg_XorExpressionPred,
    MyCcdfg_EqExpressionPred,
    MyCcdfg_NeqExpressionPred,
    MyCcdfg_GeExpressionPred,
    MyCcdfg_GtExpressionPred,
    MyCcdfg_LeExpressionPred,
    MyCcdfg_LtExpressionPred,
    MyCcdfg_LshiftExpressionPred,
    MyCcdfg_RshiftExpressionPred,
    MyCcdfg_AddExpressionPred,
    MyCcdfg_SubExpressionPred,
    MyCcdfg_CastExpressionPred,
    MyCcdfg_MultExpressionPred,
    MyCcdfg_DivExpressionPred,
    MyCcdfg_ModuloExpressionPred,
    MyCcdfg_SizeofExprExpressionPred,
    MyCcdfg_SizeofTypeExpressionPred,
    MyCcdfg_DerefExpressionPred,
    MyCcdfg_AddrofExpressionPred,
    MyCcdfg_NegExpressionPred,
    MyCcdfg_NotExpressionPred,
    MyCcdfg_LnotExpressionPred,
    MyCcdfg_FieldExpressionPred,
    MyCcdfg_PtrFieldExpressionPred,
    MyCcdfg_FunCallExpressionPred,
    MyCcdfg_ArrayExpressionPred,
    MyCcdfg_VarExpressionPred,
    MyCcdfg_ConstExpressionPred
]>;


def MyCcdfg_LvalueVarPred : CPred<"mlir::isa<MyCcdfg_LvalueVarType>($_self)">;
def MyCcdfg_LvalueArrayPred : CPred<"mlir::isa<MyCcdfg_LvalueArrayType>($_self)">;
def MyCcdfg_LvalueDerefPred : CPred<"mlir::isa<MyCcdfg_LvalueDerefType>($_self)">;
def MyCcdfg_LvalueFieldPred : CPred<"mlir::isa<MyCcdfg_LvalueFieldType>($_self)">;
def MyCcdfg_LvaluePtrFieldPred : CPred<"mlir::isa<MyCcdfg_LvaluePtrFieldType>($_self)">;

def MyCcdfg_LvaluePred : Or<[
    CPred<"mlir::isa<MyCcdfg_LvalueType>($_self)">,
    MyCcdfg_LvalueVarPred,
    MyCcdfg_LvalueArrayPred,
    MyCcdfg_LvalueDerefPred,
    MyCcdfg_LvalueFieldPred,
    MyCcdfg_LvaluePtrFieldPred
]>;

def MyCcdfg_BasicBlockPred : CPred<"mlir::isa<MyCcdfg_BasicBlockType>($_self)">;

def MyCcdfg_FunCallInstrPred : CPred<"mlir::isa<MyCcdfg_FunCallInstrType>($_self)">;
def MyCcdfg_AffectInstrPred : CPred<"mlir::isa<MyCcdfg_AffectInstrType>($_self)">;
def MyCcdfg_InstrPred : Or<[
    CPred<"mlir::isa<MyCcdfg_InstrType>($_self)">,
    MyCcdfg_FunCallInstrPred,
    MyCcdfg_AffectInstrPred
]>;

def MyCcdfg_InputTransitionPred : CPred<"mlir::isa<MyCcdfg_InputTransitionType>($_self)">;
def MyCcdfg_OutputTransitionPred : CPred<"mlir::isa<MyCcdfg_OutputTransitionType>($_self)">;
def MyCcdfg_UnconditionalTransitionPred : CPred<"mlir::isa<MyCcdfg_UnconditionalTransitionType>($_self)">;
def MyCcdfg_ConditionalTransitionPred : CPred<"mlir::isa<MyCcdfg_ConditionalTransitionType>($_self)">;
def MyCcdfg_TransitionPred : Or<[
    CPred<"mlir::isa<MyCcdfg_TransitionType>($_self)">,
    MyCcdfg_InputTransitionPred,
    MyCcdfg_OutputTransitionPred,
    MyCcdfg_UnconditionalTransitionPred,
    MyCcdfg_ConditionalTransitionPred
]>;



#endif // MYCCDFG_TYPES_TD__