//===---------------------MyCFrontendVisitor.cpp---------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#include "MyCFrontendVisitor.h"

std::any MyCFrontendVisitor::visitProgram(MyCParser::ProgramContext *context) {
  VisitRes result;

  VisitRes itemsRes;

  for (auto *item : context->items)
    itemsRes += std::any_cast<VisitRes>(visitProgramItem(item));

  result.program = itemsRes.program;
  int programIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(programIndex);
  result.types = "!myCast.program";

  result.program += result.vars + " = myCast.program (";
  if (!context->items.empty())
    result.program += itemsRes.vars + " : " + itemsRes.types;
  result.program += ")\n";

  return result;
}

std::any MyCFrontendVisitor::visitProgramItem(MyCParser::ProgramItemContext *context) {
  if (context->funDef() != nullptr)
    return visitFunDef(context->funDef());
  if (context->varDecl() != nullptr)
    return visitVarDecl(context->varDecl());
  return visitTypeDef(context->typeDef());
}


std::any MyCFrontendVisitor::visitFunDef(MyCParser::FunDefContext *context) {
  VisitRes result;

  VisitRes protoRes = std::any_cast<VisitRes>(visitFunProto(context->proto));
  VisitRes bodyRes;
  for (auto *body : context->body)
    bodyRes = std::any_cast<VisitRes>(visitCompoundStatement(body));

  int funDefIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(funDefIndex);
  result.types = "!myCast.funDef";

  result.program = protoRes.program + bodyRes.program;
  result.program += result.vars + " = myCast.funDef ( (" + protoRes.vars +
                    " : " + protoRes.types + "), ";
  if (!context->body.empty())
    result.program += "(" + bodyRes.vars + " : " + bodyRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitFunProto(MyCParser::FunProtoContext *context) {
  VisitRes result;

  VisitRes returnTypeRes =
      std::any_cast<VisitRes>(visitType(context->returnType));
  VisitRes argsRes;
  for (auto *arg : context->args)
    argsRes += std::any_cast<VisitRes>(visitArgument(arg));

  int funProtoIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(funProtoIndex);
  result.types = "!myCast.funProto";

  result.program = returnTypeRes.program + argsRes.program;
  result.program += result.vars + " = myCast.funProto ( (" +
                    returnTypeRes.vars + " : " + returnTypeRes.types + "), " +
                    context->name->getText() + ", ";
  if (!context->args.empty())
    result.program += "(" + argsRes.vars + " : " + argsRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitArgument(MyCParser::ArgumentContext *context) {
  VisitRes result;

  VisitRes typeRes = std::any_cast<VisitRes>(visitType(context->type_));
  VisitRes sizesRes;

  for (auto *size : context->sizes)
    sizesRes += std::any_cast<VisitRes>(visitExpression(size));

  int argumentIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(argumentIndex);
  result.types = "!myCast.argument";

  result.program = typeRes.program + sizesRes.program;
  result.program += result.vars + " = myCast.argument ( (" + typeRes.vars +
                    " : " + typeRes.types + "), " + context->name->getText() +
                    ", ";
  if (!context->sizes.empty())
    result.program += "(" + sizesRes.vars + " : " + sizesRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any MyCFrontendVisitor::visitVarDecl(MyCParser::VarDeclContext *context) {
  VisitRes result;

  VisitRes typeRes = std::any_cast<VisitRes>(visitType(context->type_));
  VisitRes declsRes;
  for (auto *decl : context->decls)
    declsRes += std::any_cast<VisitRes>(visitBaseVarDecl(decl));

  int varDeclIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(varDeclIndex);
  result.types = "!myCast.varDecl";

  result.program = typeRes.program + declsRes.program;
  result.program += result.vars + " = myCast.varDecl ( (" + typeRes.vars +
                    " : " + typeRes.types + "), ";
  if (!context->decls.empty())
    result.program += "(" + declsRes.vars + " : " + declsRes.types + ")";

  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitBaseVarDecl(MyCParser::BaseVarDeclContext *context) {
  VisitRes result;

  VisitRes sizesRes;
  VisitRes valueRes;

  for (auto *size : context->sizes)
    sizesRes += std::any_cast<VisitRes>(visitExpression(size));
  for (auto *value : context->value)
    valueRes += std::any_cast<VisitRes>(visitAssignmentExpression(value));

  int baseVarDeclIndex = getNextVarIndex();
  result.vars = "%" + std::to_string(baseVarDeclIndex);
  result.types = "!myCast.baseVarDecl";
  result.program = sizesRes.program + valueRes.program;

  result.program +=
      result.vars + " = myCast.baseVarDecl (" + context->name->getText() + ", ";
  if (!context->sizes.empty())
    result.program += "(" + sizesRes.vars + " : " + sizesRes.types + ")";
  result.program += ", ";
  if (!context->value.empty())
    result.program += "(" + valueRes.vars + " : " + valueRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any MyCFrontendVisitor::visitTypeDef(MyCParser::TypeDefContext *context) {
  if (auto *unionDefContext =
          dynamic_cast<MyCParser::UnionDefContext *>(context))
    return visitUnionDef(unionDefContext);

  if (auto *structDefContext =
          dynamic_cast<MyCParser::StructDefContext *>(context))
    return visitStructDef(structDefContext);

  if (auto *enumDefContext = dynamic_cast<MyCParser::EnumDefContext *>(context))
    return visitEnumDef(enumDefContext);

  if (auto *aliasDefContext =
          dynamic_cast<MyCParser::AliasDefContext *>(context))
    return visitAliasDef(aliasDefContext);

  return nullptr;
}

std::any
MyCFrontendVisitor::visitUnionDef(MyCParser::UnionDefContext *context) {
  VisitRes result;

  VisitRes declsRes;
  for (auto *decl : context->decls) {
    declsRes += std::any_cast<VisitRes>(visitVarDecl(decl));
  }

  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unionDef";
  result.program = declsRes.program;
  result.program +=
      result.vars + " = myCast.unionDef (" + context->name->getText() + +",";
  if (!context->decls.empty())
    result.program += "(" + declsRes.vars + " : " + declsRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitStructDef(MyCParser::StructDefContext *context) {
  VisitRes result;

  VisitRes declsRes;
  for (auto *decl : context->decls) {
    declsRes += std::any_cast<VisitRes>(visitVarDecl(decl));
  }

  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.structDef";
  result.program = declsRes.program;
  result.program +=
      result.vars + " = myCast.structDef (" + context->name->getText() + +",";
  if (!context->decls.empty())
    result.program += "(" + declsRes.vars + " : " + declsRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any MyCFrontendVisitor::visitEnumDef(MyCParser::EnumDefContext *context) {
  VisitRes result;

  VisitRes itemsRes;
  for (auto *item : context->items) {
    itemsRes += std::any_cast<VisitRes>(visitEnumItem(item));
  }

  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.enumDef";
  result.program = itemsRes.program;
  result.program +=
      result.vars + " = myCast.enumDef (" + context->name->getText() + +",";
  if (!context->items.empty())
    result.program += "(" + itemsRes.vars + " : " + itemsRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitAliasDef(MyCParser::AliasDefContext *context) {
  VisitRes result;

  VisitRes typeRes = std::any_cast<VisitRes>(visitType(context->bt));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.aliasDef";
  result.program = typeRes.program + result.vars + " = myCast.aliasDef ((" +
                   typeRes.vars + " : " + typeRes.types + ")," +
                   context->name->getText() + ")";

  return result;
}

std::any
MyCFrontendVisitor::visitEnumItem(MyCParser::EnumItemContext *context) {
  VisitRes result;
  
  VisitRes valueRes;
  if (context->value != nullptr)
    valueRes = std::any_cast<VisitRes>(visitConditionalExpression(context->value));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.enumItem";
  result.program = valueRes.program + result.vars + " = myCast.enumItem (" +
            context->name->getText() + ", ";
  if (context->value != nullptr)
    result.program += "(" + valueRes.vars + " : " + valueRes.types + ")";
  result.program += ")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitStatement(MyCParser::StatementContext *context) {
  if (context->labeledStatement() != nullptr)
    return visitLabeledStatement(context->labeledStatement());
  if (context->compoundStatement() != nullptr)
    return visitCompoundStatement(context->compoundStatement());
  if (context->varDeclStatement() != nullptr)
    return visitVarDeclStatement(context->varDeclStatement());
  if (context->expressionStatement() != nullptr)
    return visitExpressionStatement(context->expressionStatement());
  if (context->ifStatement() != nullptr)
    return visitIfStatement(context->ifStatement());
  if (context->switchStatement() != nullptr)
    return visitSwitchStatement(context->switchStatement());
  if (context->whileStatement() != nullptr)
    return visitWhileStatement(context->whileStatement());
  if (context->doWhileStatement() != nullptr)
    return visitDoWhileStatement(context->doWhileStatement());
  if (context->forStatement() != nullptr)
    return visitForStatement(context->forStatement());
  if (context->gotoStatement() != nullptr)
    return visitGotoStatement(context->gotoStatement());
  if (context->continueStatement() != nullptr)
    return visitContinueStatement(context->continueStatement());
  if (context->breakStatement() != nullptr)
    return visitBreakStatement(context->breakStatement());
  if (context->returnStatement() != nullptr)
    return visitReturnStatement(context->returnStatement());
  return nullptr;
}

std::any MyCFrontendVisitor::visitLabeledStatement(
    MyCParser::LabeledStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitCompoundStatement(
    MyCParser::CompoundStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitVarDeclStatement(
    MyCParser::VarDeclStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitExpressionStatement(
    MyCParser::ExpressionStatementContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitIfStatement(MyCParser::IfStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitSwitchStatement(
    MyCParser::SwitchStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitSwitchCaseItem(
    MyCParser::SwitchCaseItemContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitSwitchDefaultItem(
    MyCParser::SwitchDefaultItemContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitWhileStatement(
    MyCParser::WhileStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitDoWhileStatement(
    MyCParser::DoWhileStatementContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitForStatement(MyCParser::ForStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitGotoStatement(
    MyCParser::GotoStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitContinueStatement(
    MyCParser::ContinueStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitBreakStatement(
    MyCParser::BreakStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitReturnStatement(
    MyCParser::ReturnStatementContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitType(MyCParser::TypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitStaticTypeModifier(
    MyCParser::StaticTypeModifierContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitConstTypeModifier(
    MyCParser::ConstTypeModifierContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitExternTypeModifier(
    MyCParser::ExternTypeModifierContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitVolatileTypeModifier(
    MyCParser::VolatileTypeModifierContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitVoidType(MyCParser::VoidTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnsignedLongLongType(
    MyCParser::UnsignedLongLongTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnsignedLongType(
    MyCParser::UnsignedLongTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnsignedShortType(
    MyCParser::UnsignedShortTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnsignedCharType(
    MyCParser::UnsignedCharTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnsignedIntType(
    MyCParser::UnsignedIntTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitCharType(MyCParser::CharTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitShortType(MyCParser::ShortTypeContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitIntType(MyCParser::IntTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitLongLongType(MyCParser::LongLongTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitLongType(MyCParser::LongTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitFloatType(MyCParser::FloatTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitDoubleType(MyCParser::DoubleTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitEnumType(MyCParser::EnumTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitStructType(MyCParser::StructTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitUnionType(MyCParser::UnionTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitAliasType(MyCParser::AliasTypeContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitExpression(MyCParser::ExpressionContext *context) {
  return visitAssignmentExpression(context->assignmentExpression());
}

std::any MyCFrontendVisitor::visitAssignmentExpression(
    MyCParser::AssignmentExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitEqOp(MyCParser::EqOpContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitStarEqOp(MyCParser::StarEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitDivEqOp(MyCParser::DivEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitModuloEqOp(MyCParser::ModuloEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitPlusEqOp(MyCParser::PlusEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitMinusEqOp(MyCParser::MinusEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitLeftShiftEqOp(
    MyCParser::LeftShiftEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitRightShiftEqOp(
    MyCParser::RightShiftEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitAndEqOp(MyCParser::AndEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitXorEqOp(MyCParser::XorEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitOrEqOp(MyCParser::OrEqOpContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitConditionalExpression(
    MyCParser::ConditionalExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitLogicalOrExpression(
    MyCParser::LogicalOrExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitLogicalAndExpression(
    MyCParser::LogicalAndExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitOrExpression(MyCParser::OrExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitXorExpression(
    MyCParser::XorExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitAndExpression(
    MyCParser::AndExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitEqualityExpression(
    MyCParser::EqualityExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitEqualOperator(
    MyCParser::EqualOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitNotEqualOperator(
    MyCParser::NotEqualOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitRelationalExpression(
    MyCParser::RelationalExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitGeOperator(MyCParser::GeOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitGtOperator(MyCParser::GtOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitLeOperator(MyCParser::LeOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitLtOperator(MyCParser::LtOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitShiftExpression(
    MyCParser::ShiftExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitLshiftOperator(
    MyCParser::LshiftOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitRshiftOperator(
    MyCParser::RshiftOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitAdditiveExpression(
    MyCParser::AdditiveExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitPlusOperator(MyCParser::PlusOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitMinusOperator(
    MyCParser::MinusOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitMultiplicativeExpression(
    MyCParser::MultiplicativeExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitMultOperator(MyCParser::MultOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitDivOperator(MyCParser::DivOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitModuloOperator(
    MyCParser::ModuloOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitCastExpression(
    MyCParser::CastExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnaryExpression(
    MyCParser::UnaryExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitUnopExpression(
    MyCParser::UnopExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitIncrOperator(MyCParser::IncrOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitDecrOperator(MyCParser::DecrOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitAddrofOperator(
    MyCParser::AddrofOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitDerefOperator(
    MyCParser::DerefOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitPositiveOperator(
    MyCParser::PositiveOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitNegativeOperator(
    MyCParser::NegativeOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitNotOperator(MyCParser::NotOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any
MyCFrontendVisitor::visitLnotOperator(MyCParser::LnotOperatorContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitSizeofExpression(
    MyCParser::SizeofExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitPrimaryPostfixExpression(
    MyCParser::PrimaryPostfixExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitPostdecrExpression(
    MyCParser::PostdecrExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitArrayExpression(
    MyCParser::ArrayExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitPostincrExpression(
    MyCParser::PostincrExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitFieldExpression(
    MyCParser::FieldExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitFunCallExpression(
    MyCParser::FunCallExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitPtrFieldExpression(
    MyCParser::PtrFieldExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitVarExpression(
    MyCParser::VarExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitIntExpression(
    MyCParser::IntExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitFloatExpression(
    MyCParser::FloatExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitParentExpression(
    MyCParser::ParentExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitStructureExpression(
    MyCParser::StructureExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitGenericExpression(
    MyCParser::GenericExpressionContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitTypeGenericItem(
    MyCParser::TypeGenericItemContext *context) {
  VisitRes result;

  return result;
}

std::any MyCFrontendVisitor::visitDefaultGenericItem(
    MyCParser::DefaultGenericItemContext *context) {
  VisitRes result;

  return result;
}