//===-----------------------MyCFrontendVisitor.h---------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCFRONTEND_H__
#define MYCFRONTEND_H__

#include <string>

#include "MyCVisitor.h"

class MyCFrontendVisitor : public antlr4::tree::AbstractParseTreeVisitor {
  int varIndex = 0;

public:
  int getNextVarIndex() { return varIndex++; }

  /**
   * Visit parse trees produced by MyCParser.
   */
  std::any visitProgram(MyCParser::ProgramContext *context);
  
  std::any visitProgramItem(MyCParser::ProgramItemContext *context);

  std::any visitFunDef(MyCParser::FunDefContext *context);

  std::any visitFunProto(MyCParser::FunProtoContext *context);

  std::any visitArgument(MyCParser::ArgumentContext *context);

  std::any visitVarDecl(MyCParser::VarDeclContext *context);

  std::any visitBaseVarDecl(MyCParser::BaseVarDeclContext *context);

  std::any visitTypeDef(MyCParser::TypeDefContext *context);

  std::any visitUnionDef(MyCParser::UnionDefContext *context);

  std::any visitStructDef(MyCParser::StructDefContext *context);

  std::any visitEnumDef(MyCParser::EnumDefContext *context);

  std::any visitAliasDef(MyCParser::AliasDefContext *context);

  std::any visitEnumItem(MyCParser::EnumItemContext *context);

  std::any visitStatement(MyCParser::StatementContext *context);

  std::any visitLabeledStatement(MyCParser::LabeledStatementContext *context);

  std::any visitCompoundStatement(MyCParser::CompoundStatementContext *context);

  std::any visitVarDeclStatement(MyCParser::VarDeclStatementContext *context);

  std::any
  visitExpressionStatement(MyCParser::ExpressionStatementContext *context);

  std::any visitIfStatement(MyCParser::IfStatementContext *context);

  std::any visitSwitchStatement(MyCParser::SwitchStatementContext *context);

  std::any visitSwitchCaseItem(MyCParser::SwitchCaseItemContext *context);

  std::any visitSwitchDefaultItem(MyCParser::SwitchDefaultItemContext *context);

  std::any visitWhileStatement(MyCParser::WhileStatementContext *context);

  std::any visitDoWhileStatement(MyCParser::DoWhileStatementContext *context);

  std::any visitForStatement(MyCParser::ForStatementContext *context);

  std::any visitGotoStatement(MyCParser::GotoStatementContext *context);

  std::any visitContinueStatement(MyCParser::ContinueStatementContext *context);

  std::any visitBreakStatement(MyCParser::BreakStatementContext *context);

  std::any visitReturnStatement(MyCParser::ReturnStatementContext *context);

  std::any visitType(MyCParser::TypeContext *context);

  std::any
  visitStaticTypeModifier(MyCParser::StaticTypeModifierContext *context);

  std::any visitConstTypeModifier(MyCParser::ConstTypeModifierContext *context);

  std::any
  visitExternTypeModifier(MyCParser::ExternTypeModifierContext *context);

  std::any
  visitVolatileTypeModifier(MyCParser::VolatileTypeModifierContext *context);

  std::any visitVoidType(MyCParser::VoidTypeContext *context);

  std::any
  visitUnsignedLongLongType(MyCParser::UnsignedLongLongTypeContext *context);

  std::any visitUnsignedLongType(MyCParser::UnsignedLongTypeContext *context);

  std::any visitUnsignedShortType(MyCParser::UnsignedShortTypeContext *context);

  std::any visitUnsignedCharType(MyCParser::UnsignedCharTypeContext *context);

  std::any visitUnsignedIntType(MyCParser::UnsignedIntTypeContext *context);

  std::any visitCharType(MyCParser::CharTypeContext *context);

  std::any visitShortType(MyCParser::ShortTypeContext *context);

  std::any visitIntType(MyCParser::IntTypeContext *context);

  std::any visitLongLongType(MyCParser::LongLongTypeContext *context);

  std::any visitLongType(MyCParser::LongTypeContext *context);

  std::any visitFloatType(MyCParser::FloatTypeContext *context);

  std::any visitDoubleType(MyCParser::DoubleTypeContext *context);

  std::any visitEnumType(MyCParser::EnumTypeContext *context);

  std::any visitStructType(MyCParser::StructTypeContext *context);

  std::any visitUnionType(MyCParser::UnionTypeContext *context);

  std::any visitAliasType(MyCParser::AliasTypeContext *context);

  std::any visitExpression(MyCParser::ExpressionContext *context);

  std::any
  visitAssignmentExpression(MyCParser::AssignmentExpressionContext *context);

  std::any visitEqOp(MyCParser::EqOpContext *context);

  std::any visitStarEqOp(MyCParser::StarEqOpContext *context);

  std::any visitDivEqOp(MyCParser::DivEqOpContext *context);

  std::any visitModuloEqOp(MyCParser::ModuloEqOpContext *context);

  std::any visitPlusEqOp(MyCParser::PlusEqOpContext *context);

  std::any visitMinusEqOp(MyCParser::MinusEqOpContext *context);

  std::any visitLeftShiftEqOp(MyCParser::LeftShiftEqOpContext *context);

  std::any visitRightShiftEqOp(MyCParser::RightShiftEqOpContext *context);

  std::any visitAndEqOp(MyCParser::AndEqOpContext *context);

  std::any visitXorEqOp(MyCParser::XorEqOpContext *context);

  std::any visitOrEqOp(MyCParser::OrEqOpContext *context);

  std::any
  visitConditionalExpression(MyCParser::ConditionalExpressionContext *context);

  std::any
  visitLogicalOrExpression(MyCParser::LogicalOrExpressionContext *context);

  std::any
  visitLogicalAndExpression(MyCParser::LogicalAndExpressionContext *context);

  std::any visitOrExpression(MyCParser::OrExpressionContext *context);

  std::any visitXorExpression(MyCParser::XorExpressionContext *context);

  std::any visitAndExpression(MyCParser::AndExpressionContext *context);

  std::any
  visitEqualityExpression(MyCParser::EqualityExpressionContext *context);

  std::any visitEqualOperator(MyCParser::EqualOperatorContext *context);

  std::any visitNotEqualOperator(MyCParser::NotEqualOperatorContext *context);

  std::any
  visitRelationalExpression(MyCParser::RelationalExpressionContext *context);

  std::any visitGeOperator(MyCParser::GeOperatorContext *context);

  std::any visitGtOperator(MyCParser::GtOperatorContext *context);

  std::any visitLeOperator(MyCParser::LeOperatorContext *context);

  std::any visitLtOperator(MyCParser::LtOperatorContext *context);

  std::any visitShiftExpression(MyCParser::ShiftExpressionContext *context);

  std::any visitLshiftOperator(MyCParser::LshiftOperatorContext *context);

  std::any visitRshiftOperator(MyCParser::RshiftOperatorContext *context);

  std::any
  visitAdditiveExpression(MyCParser::AdditiveExpressionContext *context);

  std::any visitPlusOperator(MyCParser::PlusOperatorContext *context);

  std::any visitMinusOperator(MyCParser::MinusOperatorContext *context);

  std::any visitMultiplicativeExpression(
      MyCParser::MultiplicativeExpressionContext *context);

  std::any visitMultOperator(MyCParser::MultOperatorContext *context);

  std::any visitDivOperator(MyCParser::DivOperatorContext *context);

  std::any visitModuloOperator(MyCParser::ModuloOperatorContext *context);

  std::any visitCastExpression(MyCParser::CastExpressionContext *context);

  std::any visitUnaryExpression(MyCParser::UnaryExpressionContext *context);

  std::any visitUnopExpression(MyCParser::UnopExpressionContext *context);

  std::any visitIncrOperator(MyCParser::IncrOperatorContext *context);

  std::any visitDecrOperator(MyCParser::DecrOperatorContext *context);

  std::any visitAddrofOperator(MyCParser::AddrofOperatorContext *context);

  std::any visitDerefOperator(MyCParser::DerefOperatorContext *context);

  std::any visitPositiveOperator(MyCParser::PositiveOperatorContext *context);

  std::any visitNegativeOperator(MyCParser::NegativeOperatorContext *context);

  std::any visitNotOperator(MyCParser::NotOperatorContext *context);

  std::any visitLnotOperator(MyCParser::LnotOperatorContext *context);

  std::any visitSizeofExpression(MyCParser::SizeofExpressionContext *context);

  std::any visitPrimaryPostfixExpression(
      MyCParser::PrimaryPostfixExpressionContext *context);

  std::any
  visitPostdecrExpression(MyCParser::PostdecrExpressionContext *context);

  std::any visitArrayExpression(MyCParser::ArrayExpressionContext *context);

  std::any
  visitPostincrExpression(MyCParser::PostincrExpressionContext *context);

  std::any visitFieldExpression(MyCParser::FieldExpressionContext *context);

  std::any visitFunCallExpression(MyCParser::FunCallExpressionContext *context);

  std::any
  visitPtrFieldExpression(MyCParser::PtrFieldExpressionContext *context);

  std::any visitVarExpression(MyCParser::VarExpressionContext *context);

  std::any visitIntExpression(MyCParser::IntExpressionContext *context);

  std::any visitFloatExpression(MyCParser::FloatExpressionContext *context);

  std::any visitParentExpression(MyCParser::ParentExpressionContext *context);

  std::any
  visitStructureExpression(MyCParser::StructureExpressionContext *context);

  std::any visitGenericExpression(MyCParser::GenericExpressionContext *context);

  std::any visitTypeGenericItem(MyCParser::TypeGenericItemContext *context);

  std::any
  visitDefaultGenericItem(MyCParser::DefaultGenericItemContext *context);
};

class VisitRes {
public:
  std::string vars;
  std::string types;
  std::string program;
  VisitRes() {
    vars = "";
    types = "";
    program = "";
  }

  VisitRes(std::string vars, std::string types, std::string program)
      : vars(vars), types(types), program(program) {}

  VisitRes operator+(const VisitRes &other) const {
    return VisitRes((this->vars.compare("") == 0) ? other.vars : (this->vars + ", " + other.vars),
                    (this->vars.compare("") == 0) ? other.types : (this->types + ", " + other.types),
                    this->program + other.program);
  }

  VisitRes &operator+=(const VisitRes &other) {
    if (this->vars.compare("") == 0) {
      this->vars = other.vars;
      this->types = other.types;
    } else {
      this->vars += ", " + other.vars;
      this->types += ", " + other.types;
    }
    this->program += other.program;
    return *this;
  }
};

#endif