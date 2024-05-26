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
        $items attr-dict `:` type($items)
    }];
}

def Cast_FunDefOp : Cast_Op<"funDef"> {
    let arguments = (ins Type<Cast_FunProtoPred>:$proto,
                         Type<Cast_CompoundStatementPred>:$body);
    let results = (outs Cast_FunDefNodeType:$res);
    let assemblyFormat = [{
        $proto `:` type($proto) attr-dict $body `:` type($body)
    }];
}

def Cast_FunProtoOp : Cast_Op<"funProto"> {
    let arguments = (ins I1:$isInlined,
                         Type<Cast_TypePred>:$returnType,
                         I32:$stars,
                         Variadic<Type<Cast_ArgumentPred>>:$args
                    );
    let results = (outs Cast_FunProtoNodeType:$res);
    let assemblyFormat = [{
        $isInlined $returnType `:` type($returnType) $stars attr-dict $args `:` type($args)
    }];
}

def Cast_ArgumentOp : Cast_Op<"argument"> {
    let arguments = (ins Type<Cast_TypePred>:$varType,
                         I32:$stars,
                         Builtin_StringAttr:$name,
                         Variadic<Type<Cast_ExpressionPred>>:$sizes);
    let results = (outs Cast_ArgumentNodeType:$res);
    let assemblyFormat = [{
        $varType `:` type($varType) $stars $name attr-dict $sizes `:` type($sizes)
    }];
}

def Cast_VarDeclOp : Cast_Op<"varDecl"> {
    let arguments = (ins Type<Cast_TypePred>:$varType,
                         Variadic<Type<Cast_BaseVarDeclPred>>:$decls);
    let results = (outs Cast_VarDeclNodeType:$res);
    let assemblyFormat = [{
        $varType `:` type($varType) attr-dict $decls `:` type($decls)
    }];
}

def Cast_BaseVarDeclOp : Cast_Op<"baseVarDecl"> {
    let arguments = (ins I32:$stars,
                         Builtin_StringAttr:$name,
                         Variadic<Type<Cast_ExpressionPred>>:$sizes);
    let results = (outs Cast_BaseVarDeclNodeType);
    let assemblyFormat = [{
        $stars $name attr-dict $sizes `:` type($sizes)
    }];
}

def Cast_UnionDefOp : Cast_Op<"unionDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_VarDeclPred>>:$decls);
    let results = (outs Cast_UnionDefNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict $decls `:` type($decls)
    }];
}

def Cast_StructDefOp : Cast_Op<"structDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_VarDeclPred>>:$decls);
    let results = (outs Cast_StructDefNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict $decls `:` type($decls)
    }];
}

def Cast_EnumDefOp : Cast_Op<"enumDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Variadic<Type<Cast_EnumItemPred>>:$items);
    let results = (outs Cast_EnumDefNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict $items `:` type($items)
    }];
}

def Cast_AliasDefOp : Cast_Op<"aliasDef"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Type<Cast_TypePred>:$baseType);
    let results = (outs Cast_AliasDefNodeType:$results);
    let assemblyFormat = [{
        $name attr-dict $baseType `:` type($baseType)
    }];
}

def Cast_EnumItemOp : Cast_Op<"enumItem"> {
    let arguments = (ins Builtin_StringAttr:$name,
                         Optional<Type<Cast_ConditionalExpressionPred>>:$value);
    let results = (outs Cast_EnumItemNodeType:$res);
    let assemblyFormat = [{
        $name attr-dict $value `:` type($value)
    }];
}

def Cast_LabeledStatementOp : Cast_Op<"labeledStatement"> {
    let arguments = (ins Builtin_StringAttr:$label,
                         Type<Cast_StatementPred>:$stmt);
    let results = (outs Cast_LabeledStatementNodeType:$res);
    let assemblyFormat = [{
        $label attr-dict $stmt `:` type($stmt)
    }];
}

def Cast_CompoundStatmentOp : Cast_Op<"compoundStatement"> {
    let arguments = (ins Variadic<Type<Cast_StatementPred>>:$stmt);
    let results = (outs Cast_CompoundStatementNodeType:$res);
    let assemblyFormat = [{
        attr-dict $stmt `:` type($stmt)
    }];
}

def Cast_VarDeclStatementOp : Cast_Op<"varDecl"> {
    let arguments = (ins Type<Cast_VarDeclPred>:$decl);
    let results = (outs Cast_VarDeclStatementNodeType:$res);
    let assemblyFormat = [{
        attr-dict $decl `:` type($decl)
    }];
}

def Cast_ExpressionStatementOp : Cast_Op<"expressionStatement"> {
    let arguments = (ins Optional<Type<Cast_ExpressionPred>>:$value);
    let results = (outs Cast_ExpressionStatementNodeType:$res);
    let assemblyFormat = [{
        attr-dict $value `:` type($value)
    }];
}

def Cast_IfStatementOp : Cast_Op<"ifStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$thenPart,
                         Optional<Type<Cast_StatementPred>>:$elsePart);
    let results = (outs Cast_IfStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) attr-dict $thenPart `:` type($thenPart) `,` $elsePart `:` type($elsePart)
    }];
}

def Cast_SwitchStatementOp : Cast_Op<"switchStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Variadic<Type<Cast_SwitchItemPred>>:$items);
    let results = (outs Cast_SwitchStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) attr-dict $items `:` type($items)
    }];
}

def Cast_SwitchCaseItemOp : Cast_Op<"SwitchCaseItem"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_SwitchCaseItemNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) attr-dict $body `:` type($body)
    }];
}

def Cast_SwitchDefaultItemOp : Cast_Op<"switchDefaultItem"> {
    let arguments = (ins Type<Cast_StatementPred>:$body);
    let results = (outs Cast_SwitchDefaultItemNodeType:$res);
    let assemblyFormat = [{
        attr-dict $body `:` type($body)
    }];
}

def Cast_WhileStatementOp : Cast_Op<"whileStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_WhileStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) attr-dict $body `:` type($body)
    }];
}

def Cast_DoWhileStatementOp : Cast_Op<"doWhileStatement"> {
    let arguments = (ins Type<Cast_ExpressionPred>:$cond,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_DoWhileStatementNodeType:$res);
    let assemblyFormat = [{
        $cond `:` type($cond) attr-dict $body `:` type($body)
    }];
}

def Cast_ForStatementOp : Cast_Op<"forStatement", [AttrSizedOperandSegments]> {
    let arguments = (ins Optional<Type<Cast_VarDeclPred>>:$init,
                         Optional<Type<Cast_ExpressionPred>>:$cond,
                         Optional<Type<Cast_ExpressionPred>>:$step,
                         Type<Cast_StatementPred>:$body);
    let results = (outs Cast_ForStatementNodeType:$res);
    let assemblyFormat = [{
        $init `:` type($init) `,` $cond `:` type($cond) `,` $step `:` type($step) attr-dict $body `:` type($body)
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
        attr-dict $value `:` type($value)
    }];
}

def Cast_TypeOp : Cast_Op<"Type"> {
    let arguments = (ins Variadic<Type<Cast_TypeModifierPred>>:$modifiers,
                         Type<Cast_BaseTypePred>:$baseType);
    let results = (outs Cast_TypeNodeType:$res);
    let assemblyFormat = [{
        $modifiers `:` type($modifiers) attr-dict $baseType `:` type($baseType)
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
        attr-dict $expr `:` type($expr)
    }];
}

def Cast_AssignmentExpressionOp : Cast_Op<"AssignmentExpression"> {
    let arguments = (ins Type<Cast_UnaryExpressionPred>:$lvalue,
                         Type<Cast_AssignOperatorPred>:$op,
                         Type<Cast_AssignmentExpressionPred>:$rvalue);
    let results = (outs Cast_AssignOpExpressionNodeType:$res);
    let assemblyFormat = [{
        $lvalue `:` type($lvalue) attr-dict $op `:` type($op) $rvalue `:` type($rvalue)
    }];
}

/*

def Cast_Op : Cast_Op<""> {
    let arguments = (ins );
    let results = (outs Cast_NodeType:$res);
    let assemblyFormat = [{

    }];
}
*/
#endif // CAST_OPS