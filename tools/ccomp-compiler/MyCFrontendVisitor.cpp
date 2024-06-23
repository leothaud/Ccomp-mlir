//===---------------------MyCFrontendVisitor.cpp---------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#include "MyCParser.h"
#include "MyCFrontendVisitor.h"
#include <any>
#include <string>

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

std::any
MyCFrontendVisitor::visitProgramItem(MyCParser::ProgramItemContext *context) {
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
                    returnTypeRes.vars + " : " + returnTypeRes.types + "), \"" +
                    context->name->getText() + "\", ";
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
                    " : " + typeRes.types + "), \"" + context->name->getText() +
                    "\", ";
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
      result.vars + " = myCast.baseVarDecl (\"" + context->name->getText() + "\", ";
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
      result.vars + " = myCast.unionDef (\"" + context->name->getText() + +"\",";
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
      result.vars + " = myCast.structDef (\"" + context->name->getText() + +"\",";
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
      result.vars + " = myCast.enumDef (\"" + context->name->getText() + +"\",";
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
                   typeRes.vars + " : " + typeRes.types + "),\"" +
                   context->name->getText() + "\")\n";

  return result;
}

std::any
MyCFrontendVisitor::visitEnumItem(MyCParser::EnumItemContext *context) {
  VisitRes result;

  VisitRes valueRes;
  if (context->value != nullptr)
    valueRes =
        std::any_cast<VisitRes>(visitConditionalExpression(context->value));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.enumItem";
  result.program = valueRes.program + result.vars + " = myCast.enumItem (\"" +
                   context->name->getText() + "\", ";
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
  VisitRes stmtRes = std::any_cast<VisitRes>(visitStatement(context->stmt));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.labeledStatement";
  result.program = stmtRes.program + result.vars +
                   " = myCast.labeledStatemet (\"" + context->label->getText() +
                   "\", (" + stmtRes.vars + " : " + stmtRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitCompoundStatement(
    MyCParser::CompoundStatementContext *context) {
  VisitRes result;
  VisitRes stmtRes;
  for (auto *stmt : context->stmt)
    stmtRes += std::any_cast<VisitRes>(visitStatement(stmt));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.compoundStatement";
  result.program =
      stmtRes.program + result.vars + " = myCast.compoundStatement (";
  if (!context->stmt.empty())
    result.program += stmtRes.vars + " : " + stmtRes.types;
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitVarDeclStatement(
    MyCParser::VarDeclStatementContext *context) {
  VisitRes result;
  VisitRes varDeclStmtRes =
      std::any_cast<VisitRes>(visitVarDecl(context->decl));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.varDeclStatement";
  result.program = varDeclStmtRes.program + result.vars +
                   " = myCast.varDeclStatement (" + varDeclStmtRes.vars +
                   " : " + varDeclStmtRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitExpressionStatement(
    MyCParser::ExpressionStatementContext *context) {
  VisitRes result;
  VisitRes valueRes;
  if (context->value != nullptr)
    valueRes = std::any_cast<VisitRes>(visitExpression(context->value));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.expressionStatement";
  result.program =
      valueRes.program + result.vars + " = myCast.expressionStatement (";
  if (context->value != nullptr)
    result.program += valueRes.vars + " : " + valueRes.types;
  result.program += ")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitIfStatement(MyCParser::IfStatementContext *context) {
  VisitRes result;
  VisitRes condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  VisitRes thenRes = std::any_cast<VisitRes>(visitStatement(context->thenPart));
  VisitRes elseRes;
  if (context->elsePart != nullptr)
    elseRes = std::any_cast<VisitRes>(visitStatement(context->elsePart));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.ifStatement";
  result.program = condRes.program + thenRes.program + elseRes.program +
                   result.vars + " = myCast.ifStatement ((" + condRes.vars +
                   " : " + condRes.types + "), (" + thenRes.vars + " : " +
                   thenRes.types + "), ";
  if (context->elsePart != nullptr)
    result.program += "(" + elseRes.vars + " : " + elseRes.types + ")";
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitSwitchStatement(
    MyCParser::SwitchStatementContext *context) {
  VisitRes result;
  VisitRes condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  VisitRes itemsRes;
  for (auto *item : context->items)
    itemsRes += std::any_cast<VisitRes>(visitSwitchItem(item));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.switchStatement";
  result.program = condRes.program + itemsRes.program + result.vars +
                   " = myCast.switchStatement((" + condRes.vars + " : " +
                   condRes.types + "), ";
  if (!context->items.empty())
    result.program += "(" + itemsRes.vars + " : " + itemsRes.types + ")";
  result.program += ")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitSwitchItem(MyCParser::SwitchItemContext *context) {
  if (auto *caseItemContext =
          dynamic_cast<MyCParser::SwitchCaseItemContext *>(context))
    return visitSwitchCaseItem(caseItemContext);
  if (auto *defaultItemContext =
          dynamic_cast<MyCParser::SwitchDefaultItemContext *>(context))
    return visitSwitchDefaultItem(defaultItemContext);
  return nullptr;
}

std::any MyCFrontendVisitor::visitSwitchCaseItem(
    MyCParser::SwitchCaseItemContext *context) {
  VisitRes result;
  VisitRes condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  VisitRes bodyRes = std::any_cast<VisitRes>(visitStatement(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.switchCaseItem";
  result.program = condRes.program + bodyRes.program + result.vars +
                   " = myCast.switchCaseItem ((" + condRes.vars + " : " +
                   condRes.types + "), (" + bodyRes.vars + " : " +
                   bodyRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitSwitchDefaultItem(
    MyCParser::SwitchDefaultItemContext *context) {
  VisitRes result;
  VisitRes bodyRes = std::any_cast<VisitRes>(visitStatement(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.switchDefaultItem";
  result.program = bodyRes.program + result.vars +
                   " = myCast.switchDefaultItem (" + bodyRes.vars + " : " +
                   bodyRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitWhileStatement(
    MyCParser::WhileStatementContext *context) {
  VisitRes result;
  VisitRes condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  VisitRes bodyRes = std::any_cast<VisitRes>(visitStatement(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.whileStatement";
  result.program = condRes.program + bodyRes.program + result.vars +
                   " = myCast.whileStatement ((" + condRes.vars + " : " +
                   condRes.types + "), (" + bodyRes.vars + " : " +
                   bodyRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitDoWhileStatement(
    MyCParser::DoWhileStatementContext *context) {
  VisitRes result;
  VisitRes condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  VisitRes bodyRes = std::any_cast<VisitRes>(visitStatement(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.doWhileStatement";
  result.program = condRes.program + bodyRes.program + result.vars +
                   " = myCast.doWhileStatement ((" + condRes.vars + " : " +
                   condRes.types + "), (" + bodyRes.vars + " : " +
                   bodyRes.types + "))\n";
  return result;
}

std::any
MyCFrontendVisitor::visitForStatement(MyCParser::ForStatementContext *context) {
  VisitRes result;
  VisitRes initRes, condRes, stepRes;
  if (context->init != nullptr)
    initRes = std::any_cast<VisitRes>(visitVarDecl(context->init));
  if (context->cond != nullptr)
    condRes = std::any_cast<VisitRes>(visitExpression(context->cond));
  if (context->step != nullptr)
    stepRes = std::any_cast<VisitRes>(visitExpression(context->step));
  VisitRes bodyRes = std::any_cast<VisitRes>(visitStatement(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.forStatement";
  result.program = initRes.program + condRes.program + stepRes.program +
                   bodyRes.program + result.vars + " = myCast.forStatement(";
  if (context->init != nullptr)
    result.program += "(" + initRes.vars + " : " + initRes.types + ")";
  result.program += ", ";
  if (context->cond != nullptr)
    result.program += "(" + condRes.vars + " : " + condRes.types + ")";
  result.program += ", ";
  if (context->step != nullptr)
    result.program += "(" + stepRes.vars + " : " + stepRes.types + ")";
  result.program += ", (" + bodyRes.vars + " : " + bodyRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitGotoStatement(
    MyCParser::GotoStatementContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.gotoStatement";
  result.program = result.vars + " = myCast.gotoStatement (\"" +
                   context->label->getText() + "\")\n";
  return result;
}

std::any MyCFrontendVisitor::visitContinueStatement(
    MyCParser::ContinueStatementContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.continueStatement";
  result.program = result.vars + " = myCast.continueStatement \n";
  return result;
}

std::any MyCFrontendVisitor::visitBreakStatement(
    MyCParser::BreakStatementContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.breakStatement";
  result.program = result.vars + " = myCast.breakStatement \n";
  return result;
}

std::any MyCFrontendVisitor::visitReturnStatement(
    MyCParser::ReturnStatementContext *context) {
  VisitRes result;
  VisitRes valueRes;
  if (context->value != nullptr)
    valueRes = std::any_cast<VisitRes>(visitExpression(context->value));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.returnStatement";
  result.program =
      valueRes.program + result.vars + " = myCast.returnStatement (";
  if (context->value != nullptr)
    result.program += valueRes.vars + " : " + valueRes.types;
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitType(MyCParser::TypeContext *context) {
  if (context->child != nullptr)
    return visitBaseType(context->child);
  VisitRes result;
  VisitRes modifiersRes;
  for (auto *modifier : context->modifiers)
    modifiersRes += std::any_cast<VisitRes>(visitTypeModifier(modifier));
  VisitRes btRes = std::any_cast<VisitRes>(visitBaseType(context->bt));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.type";
  result.program =
      modifiersRes.program + btRes.program + result.vars + " = myCast.type (";
  if (!context->modifiers.empty())
    result.program +=
        "(" + modifiersRes.vars + " : " + modifiersRes.types + ")";
  result.program += ", (" + btRes.vars + " : " + btRes.types + "), " +
                    std::to_string(context->stars.size()) + ")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitTypeModifier(MyCParser::TypeModifierContext *context) {
  if (auto *staticContext =
          dynamic_cast<MyCParser::StaticTypeModifierContext *>(context))
    return visitStaticTypeModifier(staticContext);
  if (auto *constContext =
          dynamic_cast<MyCParser::ConstTypeModifierContext *>(context))
    return visitConstTypeModifier(constContext);
  if (auto *externContext =
          dynamic_cast<MyCParser::ExternTypeModifierContext *>(context))
    return visitExternTypeModifier(externContext);
  if (auto *volatileContext =
          dynamic_cast<MyCParser::VolatileTypeModifierContext *>(context))
    return visitVolatileTypeModifier(volatileContext);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitStaticTypeModifier(
    MyCParser::StaticTypeModifierContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.staticTypeModifier";
  result.program = result.vars + " = myCast.staticTypeModifier\n";
  return result;
}

std::any MyCFrontendVisitor::visitConstTypeModifier(
    MyCParser::ConstTypeModifierContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.constTypeModifier";
  result.program = result.vars + " = myCast.constTypeModifier\n";
  return result;
}

std::any MyCFrontendVisitor::visitExternTypeModifier(
    MyCParser::ExternTypeModifierContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.externTypeModifier";
  result.program = result.vars + " = myCast.externTypeModifier\n";
  return result;
}

std::any MyCFrontendVisitor::visitVolatileTypeModifier(
    MyCParser::VolatileTypeModifierContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.volatileTypeModifier";
  result.program = result.vars + " = myCast.volatileTypeModifier\n";
  return result;
}

std::any
MyCFrontendVisitor::visitBaseType(MyCParser::BaseTypeContext *context) {
  if (auto *voidTypeContext =
          dynamic_cast<MyCParser::VoidTypeContext *>(context))
    return visitVoidType(voidTypeContext);
  if (auto *unsignedLongLongTypeContext =
          dynamic_cast<MyCParser::UnsignedLongLongTypeContext *>(context))
    return visitUnsignedLongLongType(unsignedLongLongTypeContext);
  if (auto *unsignedLongTypeContext =
          dynamic_cast<MyCParser::UnsignedLongTypeContext *>(context))
    return visitUnsignedLongType(unsignedLongTypeContext);
  if (auto *unsignedShortTypeContext =
          dynamic_cast<MyCParser::UnsignedShortTypeContext *>(context))
    return visitUnsignedShortType(unsignedShortTypeContext);
  if (auto *unsignedCharTypeContext =
          dynamic_cast<MyCParser::UnsignedCharTypeContext *>(context))
    return visitUnsignedCharType(unsignedCharTypeContext);
  if (auto *unsignedIntTypeContext =
          dynamic_cast<MyCParser::UnsignedIntTypeContext *>(context))
    return visitUnsignedIntType(unsignedIntTypeContext);
  if (auto *charTypeContext =
          dynamic_cast<MyCParser::CharTypeContext *>(context))
    return visitCharType(charTypeContext);
  if (auto *shortTypeContext =
          dynamic_cast<MyCParser::ShortTypeContext *>(context))
    return visitShortType(shortTypeContext);
  if (auto *intTypeContext = dynamic_cast<MyCParser::IntTypeContext *>(context))
    return visitIntType(intTypeContext);
  if (auto *longLongTypeContext =
          dynamic_cast<MyCParser::LongLongTypeContext *>(context))
    return visitLongLongType(longLongTypeContext);
  if (auto *longTypeContext =
          dynamic_cast<MyCParser::LongTypeContext *>(context))
    return visitLongType(longTypeContext);
  if (auto *floatTypeContext =
          dynamic_cast<MyCParser::FloatTypeContext *>(context))
    return visitFloatType(floatTypeContext);
  if (auto *doubleTypeContext =
          dynamic_cast<MyCParser::DoubleTypeContext *>(context))
    return visitDoubleType(doubleTypeContext);
  if (auto *enumTypeContext =
          dynamic_cast<MyCParser::EnumTypeContext *>(context))
    return visitEnumType(enumTypeContext);
  if (auto *structTypeContext =
          dynamic_cast<MyCParser::StructTypeContext *>(context))
    return visitStructType(structTypeContext);
  if (auto *unionTypeContext =
          dynamic_cast<MyCParser::UnionTypeContext *>(context))
    return visitUnionType(unionTypeContext);
  if (auto *aliasTypeContext =
          dynamic_cast<MyCParser::AliasTypeContext *>(context))
    return visitAliasType(aliasTypeContext);
  return VisitRes();
}

std::any
MyCFrontendVisitor::visitVoidType(MyCParser::VoidTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.voidType";
  result.program = result.vars + " = myCast.voidType\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnsignedLongLongType(
    MyCParser::UnsignedLongLongTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unsignedLongLongType";
  result.program = result.vars + " = myCast.unsignedLongLongType\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnsignedLongType(
    MyCParser::UnsignedLongTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unsignedLongType";
  result.program = result.vars + " = myCast.unsignedLongType\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnsignedShortType(
    MyCParser::UnsignedShortTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unsignedShortType";
  result.program = result.vars + " = myCast.unsignedShortType\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnsignedCharType(
    MyCParser::UnsignedCharTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unsignedCharType";
  result.program = result.vars + " = myCast.unsignedCharType\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnsignedIntType(
    MyCParser::UnsignedIntTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unsignedIntType";
  result.program = result.vars + " = myCast.unsignedIntType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitCharType(MyCParser::CharTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.charType";
  result.program = result.vars + " = myCast.charType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitShortType(MyCParser::ShortTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.shortType";
  result.program = result.vars + " = myCast.shortType\n";
  return result;
}

std::any MyCFrontendVisitor::visitIntType(MyCParser::IntTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.intType";
  result.program = result.vars + " = myCast.intType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitLongLongType(MyCParser::LongLongTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.longLongType";
  result.program = result.vars + " = myCast.longLongType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitLongType(MyCParser::LongTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.longType";
  result.program = result.vars + " = myCast.longType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitFloatType(MyCParser::FloatTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.floatType";
  result.program = result.vars + " = myCast.floatType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitDoubleType(MyCParser::DoubleTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.doubleType";
  result.program = result.vars + " = myCast.doubleType\n";
  return result;
}

std::any
MyCFrontendVisitor::visitEnumType(MyCParser::EnumTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.enumType";
  result.program =
      result.vars + " = myCast.enumType (\"" + context->name->getText() + "\")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitStructType(MyCParser::StructTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.structType";
  result.program =
      result.vars + " = myCast.structType (\"" + context->name->getText() + "\")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitUnionType(MyCParser::UnionTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unionType";
  result.program =
      result.vars + " = myCast.unionType (\"" + context->name->getText() + "\")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitAliasType(MyCParser::AliasTypeContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.aliasType";
  result.program =
      result.vars + " = myCast.aliasType (\"" + context->name->getText() + "\")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitExpression(MyCParser::ExpressionContext *context) {
  return visitAssignmentExpression(context->assignmentExpression());
}

std::any MyCFrontendVisitor::visitAssignmentExpression(
    MyCParser::AssignmentExpressionContext *context) {
  if (context->child != nullptr)
    return visitConditionalExpression(context->child);
  VisitRes result;
  VisitRes lvalueRes =
      std::any_cast<VisitRes>(visitUnaryExpression(context->lvalue));
  VisitRes opRes = std::any_cast<VisitRes>(visitAssignOperator(context->op));
  VisitRes rvalueRes =
      std::any_cast<VisitRes>(visitAssignmentExpression(context->rvalue));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.assignmentExpression";
  result.program = lvalueRes.program + opRes.program + rvalueRes.program +
                   result.vars + " = myCast.assignmentExpression((" +
                   lvalueRes.vars + " : " + lvalueRes.types + "),(" +
                   opRes.vars + " : " + opRes.types + "),(" + rvalueRes.vars +
                   " : " + rvalueRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitAssignOperator(
    MyCParser::AssignOperatorContext *context) {
  if (auto *eqOpContext = dynamic_cast<MyCParser::EqOpContext *>(context))
    return visitEqOp(eqOpContext);
  if (auto *starEqOpContext =
          dynamic_cast<MyCParser::StarEqOpContext *>(context))
    return visitStarEqOp(starEqOpContext);
  if (auto *divEqOpContext = dynamic_cast<MyCParser::DivEqOpContext *>(context))
    return visitDivEqOp(divEqOpContext);
  if (auto *moduloEqOpContext =
          dynamic_cast<MyCParser::ModuloEqOpContext *>(context))
    return visitModuloEqOp(moduloEqOpContext);
  if (auto *plusEqOpContext =
          dynamic_cast<MyCParser::PlusEqOpContext *>(context))
    return visitPlusEqOp(plusEqOpContext);
  if (auto *minusEqOpContext =
          dynamic_cast<MyCParser::MinusEqOpContext *>(context))
    return visitMinusEqOp(minusEqOpContext);
  if (auto *leftShifteqOpContext =
          dynamic_cast<MyCParser::LeftShiftEqOpContext *>(context))
    return visitLeftShiftEqOp(leftShifteqOpContext);
  if (auto *rightShiftEqOpContext =
          dynamic_cast<MyCParser::RightShiftEqOpContext *>(context))
    return visitRightShiftEqOp(rightShiftEqOpContext);
  if (auto *andEqOpContext = dynamic_cast<MyCParser::AndEqOpContext *>(context))
    return visitAndEqOp(andEqOpContext);
  if (auto *xorEqOpContext = dynamic_cast<MyCParser::XorEqOpContext *>(context))
    return visitXorEqOp(xorEqOpContext);
  if (auto *orEqOpContext = dynamic_cast<MyCParser::OrEqOpContext *>(context))
    return visitOrEqOp(orEqOpContext);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitEqOp(MyCParser::EqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.eqOp";
  result.program = result.vars + " = myCast.eqOp";
  return result;
}

std::any
MyCFrontendVisitor::visitStarEqOp(MyCParser::StarEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.starEqOp";
  result.program = result.vars + " = myCast.starEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitDivEqOp(MyCParser::DivEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.divEqOp";
  result.program = result.vars + " = myCast.divEqOp";
  return result;
}

std::any
MyCFrontendVisitor::visitModuloEqOp(MyCParser::ModuloEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.moduloEqOp";
  result.program = result.vars + " = myCast.moduloEqOp";
  return result;
}

std::any
MyCFrontendVisitor::visitPlusEqOp(MyCParser::PlusEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.plusEqOp";
  result.program = result.vars + " = myCast.plusEqOp";
  return result;
}

std::any
MyCFrontendVisitor::visitMinusEqOp(MyCParser::MinusEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.minusEqOp";
  result.program = result.vars + " = myCast.minusEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitLeftShiftEqOp(
    MyCParser::LeftShiftEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.leftShiftEqOp";
  result.program = result.vars + " = myCast.leftShiftEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitRightShiftEqOp(
    MyCParser::RightShiftEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.rightShiftEqOp";
  result.program = result.vars + " = myCast.rightShiftEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitAndEqOp(MyCParser::AndEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.andEqOp";
  result.program = result.vars + " = myCast.andEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitXorEqOp(MyCParser::XorEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.xorEqOp";
  result.program = result.vars + " = myCast.xorEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitOrEqOp(MyCParser::OrEqOpContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.orEqOp";
  result.program = result.vars + " = myCast.orEqOp";
  return result;
}

std::any MyCFrontendVisitor::visitConditionalExpression(
    MyCParser::ConditionalExpressionContext *context) {
  if (context->child != nullptr)
    return visitLogicalOrExpression(context->child);
  VisitRes result;
  VisitRes condRes =
      std::any_cast<VisitRes>(visitLogicalOrExpression(context->cond));
  VisitRes thenRes =
      std::any_cast<VisitRes>(visitExpression(context->thenPart));
  VisitRes elseRes =
      std::any_cast<VisitRes>(visitConditionalExpression(context->elsePart));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.conditionalExpression";
  result.program = condRes.program + thenRes.program + elseRes.program +
                   result.vars + " = myCast.conditionalExpression((" +
                   condRes.vars + " : " + condRes.types + "),(" + thenRes.vars +
                   " : " + thenRes.types + "),(" + elseRes.vars + " : " +
                   elseRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitLogicalOrExpression(
    MyCParser::LogicalOrExpressionContext *context) {
  if (context->child != nullptr)
    return visitLogicalAndExpression(context->child);
  VisitRes result;
  VisitRes exprRes;
  for (auto *expr : context->expr)
    exprRes += std::any_cast<VisitRes>(visitLogicalAndExpression(expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.logicalOrExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.logicalOrExpression(" + exprRes.vars + " : " +
                   exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitLogicalAndExpression(
    MyCParser::LogicalAndExpressionContext *context) {
  if (context->child != nullptr)
    return visitOrExpression(context->child);
  VisitRes result;
  VisitRes exprRes;
  for (auto *expr : context->expr)
    exprRes += std::any_cast<VisitRes>(visitOrExpression(expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.logicalAndExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.logicalAndExpression(" + exprRes.vars + " : " +
                   exprRes.types + ")\n";
  return result;
}

std::any
MyCFrontendVisitor::visitOrExpression(MyCParser::OrExpressionContext *context) {
  if (context->child != nullptr)
    return visitXorExpression(context->child);
  VisitRes result;
  VisitRes exprRes;
  for (auto *expr : context->expr)
    exprRes += std::any_cast<VisitRes>(visitXorExpression(expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.orExpression";
  result.program = exprRes.program + result.vars + " = myCast.orExpression(" +
                   exprRes.vars + " : " + exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitXorExpression(
    MyCParser::XorExpressionContext *context) {
  if (context->child != nullptr)
    return visitAndExpression(context->child);
  VisitRes result;
  VisitRes exprRes;
  for (auto *expr : context->expr)
    exprRes += std::any_cast<VisitRes>(visitAndExpression(expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.xorExpression";
  result.program = exprRes.program + result.vars + " = myCast.xorExpression(" +
                   exprRes.vars + " : " + exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitAndExpression(
    MyCParser::AndExpressionContext *context) {
  if (context->child != nullptr)
    return visitEqualityExpression(context->child);
  VisitRes result;
  VisitRes exprRes;
  for (auto *expr : context->expr)
    exprRes += std::any_cast<VisitRes>(visitEqualityExpression(expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.andExpression";
  result.program = exprRes.program + result.vars + " = myCast.andExpression(" +
                   exprRes.vars + " : " + exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitEqualityExpression(
    MyCParser::EqualityExpressionContext *context) {
  if (context->child != nullptr)
    return visitRelationalExpression(context->child);
  VisitRes result;
  VisitRes lvalRes =
      std::any_cast<VisitRes>(visitEqualityExpression(context->lval));
  VisitRes opRes = std::any_cast<VisitRes>(visitEqualityOperator(context->op));
  VisitRes rvalRes =
      std::any_cast<VisitRes>(visitRelationalExpression(context->rval));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.equalityExpression";
  result.program = lvalRes.program + opRes.program + rvalRes.program +
                   result.vars + " = myCast.equalityExpression((" +
                   lvalRes.vars + " : " + lvalRes.types + "),(" + opRes.vars +
                   " : " + opRes.types + "),(" + rvalRes.vars + " : " +
                   rvalRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitEqualityOperator(
    MyCParser::EqualityOperatorContext *context) {
  if (auto *equalContext =
          dynamic_cast<MyCParser::EqualOperatorContext *>(context))
    return visitEqualOperator(equalContext);
  if (auto *notEqualContext =
          dynamic_cast<MyCParser::NotEqualOperatorContext *>(context))
    return visitNotEqualOperator(notEqualContext);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitEqualOperator(
    MyCParser::EqualOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.equalOperator";
  result.program = result.vars + " = myCast.equalOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitNotEqualOperator(
    MyCParser::NotEqualOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.notEqualOperator";
  result.program = result.vars + " = myCast.notEqualOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitRelationalExpression(
    MyCParser::RelationalExpressionContext *context) {
  if (context->child != nullptr)
    return visitShiftExpression(context->child);
  VisitRes result;
  VisitRes lvalRes =
      std::any_cast<VisitRes>(visitRelationalExpression(context->lval));
  VisitRes opRes =
      std::any_cast<VisitRes>(visitRelationalOperator(context->op));
  VisitRes rvalRes =
      std::any_cast<VisitRes>(visitShiftExpression(context->rval));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.relationalExpression";
  result.program = lvalRes.program + opRes.program + rvalRes.program +
                   result.vars + " = myCast.relationalExpression((" +
                   lvalRes.vars + " : " + lvalRes.types + "),(" + opRes.vars +
                   " : " + opRes.types + "),(" + rvalRes.vars + " : " +
                   rvalRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitRelationalOperator(
    MyCParser::RelationalOperatorContext *context) {
  if (auto *geContext = dynamic_cast<MyCParser::GeOperatorContext *>(context))
    return visitGeOperator(geContext);
  if (auto *leContext = dynamic_cast<MyCParser::LeOperatorContext *>(context))
    return visitLeOperator(leContext);
  if (auto *gtContext = dynamic_cast<MyCParser::GtOperatorContext *>(context))
    return visitGtOperator(gtContext);
  if (auto *ltContext = dynamic_cast<MyCParser::LtOperatorContext *>(context))
    return visitLtOperator(ltContext);
  return VisitRes();
}

std::any
MyCFrontendVisitor::visitGeOperator(MyCParser::GeOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.geOperator";
  result.program = result.vars + " = myCast.geOperator\n";
  return result;
}

std::any
MyCFrontendVisitor::visitGtOperator(MyCParser::GtOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.gtOperator";
  result.program = result.vars + " = myCast.gtOperator\n";
  return result;
}

std::any
MyCFrontendVisitor::visitLeOperator(MyCParser::LeOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.leOperator";
  result.program = result.vars + " = myCast.leOperator\n";
  return result;
}

std::any
MyCFrontendVisitor::visitLtOperator(MyCParser::LtOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.ltOperator";
  result.program = result.vars + " = myCast.ltOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitShiftExpression(
    MyCParser::ShiftExpressionContext *context) {
  if (context->child != nullptr)
    return visitAdditiveExpression(context->child);
  VisitRes result;
  VisitRes lvalRes =
      std::any_cast<VisitRes>(visitShiftExpression(context->lval));
  VisitRes opRes = std::any_cast<VisitRes>(visitShiftOperator(context->op));
  VisitRes rvalRes =
      std::any_cast<VisitRes>(visitAdditiveExpression(context->rval));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.shiftExpression";
  result.program = lvalRes.program + opRes.program + rvalRes.program +
                   result.vars + " = myCast.shiftExpression((" + lvalRes.vars +
                   " : " + lvalRes.types + "),(" + opRes.vars + " : " +
                   opRes.types + "),(" + rvalRes.vars + " : " + rvalRes.types +
                   "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitShiftOperator(
    MyCParser::ShiftOperatorContext *context) {
  if (auto *lshiftContext =
          dynamic_cast<MyCParser::LshiftOperatorContext *>(context))
    return visitLshiftOperator(lshiftContext);
  if (auto *rshiftContext =
          dynamic_cast<MyCParser::RshiftOperatorContext *>(context))
    return visitRshiftOperator(rshiftContext);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitLshiftOperator(
    MyCParser::LshiftOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.lshiftOperator";
  result.program = result.vars + " = myCast.lshiftOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitRshiftOperator(
    MyCParser::RshiftOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.rshiftOperator";
  result.program = result.vars + " = myCast.rshiftOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitAdditiveExpression(
    MyCParser::AdditiveExpressionContext *context) {
  if (context->child != nullptr)
    return visitMultiplicativeExpression(context->child);
  VisitRes result;
  VisitRes lvalRes =
      std::any_cast<VisitRes>(visitAdditiveExpression(context->lval));
  VisitRes opRes = std::any_cast<VisitRes>(visitAdditiveOperator(context->op));
  VisitRes rvalRes =
      std::any_cast<VisitRes>(visitMultiplicativeExpression(context->rval));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.additiveExpression";
  result.program = lvalRes.program + opRes.program + rvalRes.program +
                   result.vars + " = myCast.additiveExpression((" +
                   lvalRes.vars + " : " + lvalRes.types + "),(" + opRes.vars +
                   " : " + opRes.types + "),(" + rvalRes.vars + " : " +
                   rvalRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitAdditiveOperator(
    MyCParser::AdditiveOperatorContext *context) {
  if (auto *plusContext =
          dynamic_cast<MyCParser::PlusOperatorContext *>(context))
    return visitPlusOperator(plusContext);
  if (auto *minusContext =
          dynamic_cast<MyCParser::MinusOperatorContext *>(context))
    return visitMinusOperator(minusContext);
  return VisitRes();
}

std::any
MyCFrontendVisitor::visitPlusOperator(MyCParser::PlusOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.plusOperator";
  result.program = result.vars + " = myCast.plusOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitMinusOperator(
    MyCParser::MinusOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.minusOperator";
  result.program = result.vars + " = myCast.minusOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitMultiplicativeExpression(
    MyCParser::MultiplicativeExpressionContext *context) {
  if (context->child != nullptr)
    return visitCastExpression(context->child);
  VisitRes result;
  VisitRes lvalRes =
      std::any_cast<VisitRes>(visitMultiplicativeExpression(context->lval));
  VisitRes opRes =
      std::any_cast<VisitRes>(visitMultiplicativeOperator(context->op));
  VisitRes rvalRes =
      std::any_cast<VisitRes>(visitCastExpression(context->rval));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.multiplicativeExpression";
  result.program = lvalRes.program + opRes.program + rvalRes.program +
                   result.vars + " = myCast.multiplicativeExpression((" +
                   lvalRes.vars + " : " + lvalRes.types + "),(" + opRes.vars +
                   " : " + opRes.types + "),(" + rvalRes.vars + " : " +
                   rvalRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitMultiplicativeOperator(
    MyCParser::MultiplicativeOperatorContext *context) {
  if (auto *multContext =
          dynamic_cast<MyCParser::MultOperatorContext *>(context))
    return visitMultOperator(multContext);
  if (auto *divContext = dynamic_cast<MyCParser::DivOperatorContext *>(context))
    return visitDivOperator(divContext);
  if (auto *moduloContext =
          dynamic_cast<MyCParser::ModuloOperatorContext *>(context))
    return visitModuloOperator(moduloContext);
  return VisitRes();
}

std::any
MyCFrontendVisitor::visitMultOperator(MyCParser::MultOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.multOperator";
  result.program = result.vars + " = myCast.multOperator\n";
  return result;
}

std::any
MyCFrontendVisitor::visitDivOperator(MyCParser::DivOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.divOperator";
  result.program = result.vars + " = myCast.divOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitModuloOperator(
    MyCParser::ModuloOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.moduloOperator";
  result.program = result.vars + " = myCast.moduloOperator\n";
  return result;
}

std::any MyCFrontendVisitor::visitCastExpression(
    MyCParser::CastExpressionContext *context) {
  if (context->child != nullptr)
    return visitUnaryExpression(context->child);
  VisitRes result;
  VisitRes typeRes = std::any_cast<VisitRes>(visitType(context->newType));
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitUnaryExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.castExpression";
  result.program = typeRes.program + exprRes.program + result.vars +
                   " = myCast.castExpression((" + typeRes.vars + " : " +
                   typeRes.types + "),(" + exprRes.vars + " : " +
                   exprRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnaryExpression(
    MyCParser::UnaryExpressionContext *context) {
  if (context->postfixExpression() != nullptr)
    return visitPostfixExpression(context->postfixExpression());
  if (context->sizeofExpression() != nullptr)
    return visitSizeofExpression(context->sizeofExpression());
  return visitUnopExpression(context->unopExpression());
}

std::any MyCFrontendVisitor::visitUnopExpression(
    MyCParser::UnopExpressionContext *context) {
  VisitRes result;
  VisitRes opRes = std::any_cast<VisitRes>(visitUnaryOperator(context->op));
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitUnaryExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.unopExpression";
  result.program = opRes.program + exprRes.program + result.vars +
                   " = myCast.unopExpression((" + opRes.vars + " : " +
                   opRes.types + "),(" + exprRes.vars + " : " + exprRes.types +
                   "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitUnaryOperator(
    MyCParser::UnaryOperatorContext *context) {
  if (auto *ctx = dynamic_cast<MyCParser::IncrOperatorContext *>(context))
    return visitIncrOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::DecrOperatorContext *>(context))
    return visitDecrOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::AddrofOperatorContext *>(context))
    return visitAddrofOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::DerefOperatorContext *>(context))
    return visitDerefOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::PositiveOperatorContext *>(context))
    return visitPositiveOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::NegativeOperatorContext *>(context))
    return visitNegativeOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::NotOperatorContext *>(context))
    return visitNotOperator(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::LnotOperatorContext *>(context))
    return visitLnotOperator(ctx);
  return VisitRes();
}

std::any
MyCFrontendVisitor::visitIncrOperator(MyCParser::IncrOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.incrOperator";
  result.program = result.vars + " = myCast.incrOperator";
  return result;
}

std::any
MyCFrontendVisitor::visitDecrOperator(MyCParser::DecrOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.decrOperator";
  result.program = result.vars + " = myCast.decrOperator";
  return result;
}

std::any MyCFrontendVisitor::visitAddrofOperator(
    MyCParser::AddrofOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.addrofOperator";
  result.program = result.vars + " = myCast.addrofOperator";
  return result;
}

std::any MyCFrontendVisitor::visitDerefOperator(
    MyCParser::DerefOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.derefOperator";
  result.program = result.vars + " = myCast.derefOperator";
  return result;
}

std::any MyCFrontendVisitor::visitPositiveOperator(
    MyCParser::PositiveOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.positiveOperator";
  result.program = result.vars + " = myCast.positiveOperator";
  return result;
}

std::any MyCFrontendVisitor::visitNegativeOperator(
    MyCParser::NegativeOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.negativeOperator";
  result.program = result.vars + " = myCast.negativeOperator";
  return result;
}

std::any
MyCFrontendVisitor::visitNotOperator(MyCParser::NotOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.notOperator";
  result.program = result.vars + " = myCast.notOperator";
  return result;
}

std::any
MyCFrontendVisitor::visitLnotOperator(MyCParser::LnotOperatorContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.lnotOperator";
  result.program = result.vars + " = myCast.lnotOperator";
  return result;
}

std::any MyCFrontendVisitor::visitSizeofExpression(
    MyCParser::SizeofExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes, typeRes;
  if (context->expr != nullptr)
    exprRes = std::any_cast<VisitRes>(visitUnaryExpression(context->expr));
  if (context->type_ != nullptr)
    typeRes = std::any_cast<VisitRes>(visitType(context->type_));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.sizeofExpression";
  result.program = exprRes.program + typeRes.program + result.vars +
                   " = myCast.sizeofExpression (";
  if (context->expr != nullptr)
    result.program += "(" + exprRes.vars + " : " + exprRes.types = ")";
  result.program += ",";
  if (context->type_ != nullptr)
    result.program += "(" + typeRes.vars + " : " + typeRes.types + ")";
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitPostfixExpression(
    MyCParser::PostfixExpressionContext *context) {
  if (auto *ctx =
          dynamic_cast<MyCParser::PrimaryPostfixExpressionContext *>(context))
    return visitPrimaryPostfixExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::ArrayExpressionContext *>(context))
    return visitArrayExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::FunCallExpressionContext *>(context))
    return visitFunCallExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::FieldExpressionContext *>(context))
    return visitFieldExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::PtrFieldExpressionContext *>(context))
    return visitPtrFieldExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::PostincrExpressionContext *>(context))
    return visitPostincrExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::PostdecrExpressionContext *>(context))
    return visitPostdecrExpression(ctx);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitPrimaryPostfixExpression(
    MyCParser::PrimaryPostfixExpressionContext *context) {
  return visitPrimaryExpression(context->primaryExpression());
}

std::any MyCFrontendVisitor::visitPostdecrExpression(
    MyCParser::PostdecrExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitPostfixExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.postdecrExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.postdecrExpression (" + exprRes.vars + " : " +
                   exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitArrayExpression(
    MyCParser::ArrayExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitPostfixExpression(context->expr));
  VisitRes indexRes = std::any_cast<VisitRes>(visitExpression(context->index));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.arrayExpression";
  result.program = exprRes.program + indexRes.program + result.vars +
                   " = myCast.arrayExpression ((" + exprRes.vars + " : " +
                   exprRes.types + "),(" + indexRes.vars + " : " +
                   indexRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitPostincrExpression(
    MyCParser::PostincrExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitPostfixExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.postincrExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.postincrExpression (" + exprRes.vars + " : " +
                   exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitFieldExpression(
    MyCParser::FieldExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitPostfixExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.fieldExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.fieldExpression ((" + exprRes.vars + " : " +
                   exprRes.types + "), \"" + context->field->getText() + "\")\n";
  return result;
}

std::any MyCFrontendVisitor::visitFunCallExpression(
    MyCParser::FunCallExpressionContext *context) {
  VisitRes result;
  VisitRes argsRes;
  for (auto *arg : context->args)
    argsRes += std::any_cast<VisitRes>(visitAssignmentExpression(arg));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.funCallExpression";
  result.program = argsRes.program + result.vars +
                   " = myCast.funCallExpression (\"" + context->funName->getText() + "\",";
  if (!context->args.empty())
    result.program += "(" + argsRes.vars + " : " + argsRes.types + ")";
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitPtrFieldExpression(
    MyCParser::PtrFieldExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitPostfixExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.ptrFieldExpression";
  result.program = exprRes.program + result.vars +
                   " = myCast.ptrFieldExpression ((" + exprRes.vars + " : " +
                   exprRes.types + "), \"" + context->field->getText() + "\")\n";
  return result;
}

std::any MyCFrontendVisitor::visitPrimaryExpression(
    MyCParser::PrimaryExpressionContext *context) {
  if (auto *ctx = dynamic_cast<MyCParser::VarExpressionContext *>(context))
    return visitVarExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::IntExpressionContext *>(context))
    return visitIntExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::FloatExpressionContext *>(context))
    return visitFloatExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::StringExpressionContext *>(context))
    return visitStringExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::ParentExpressionContext *>(context))
    return visitParentExpression(ctx);
  if (auto *ctx =
          dynamic_cast<MyCParser::StructureExpressionContext *>(context))
    return visitStructureExpression(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::GenericExpressionContext *>(context))
    return visitGenericExpression(ctx);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitVarExpression(
    MyCParser::VarExpressionContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.varExpression";
  result.program = result.vars + " = myCast.varExpression(\"" +
                   context->name->getText() + "\")\n";
  return result;
}

std::any MyCFrontendVisitor::visitIntExpression(
    MyCParser::IntExpressionContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.intExpression";
  result.program = result.vars + " = myCast.intExpression (" +
                   context->val->getText() + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitFloatExpression(
    MyCParser::FloatExpressionContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.floatExpression";
  result.program = result.vars + " = myCast.floatExpression(" +
                   context->val->getText() + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitStringExpression(
    MyCParser::StringExpressionContext *context) {
  VisitRes result;
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.stringExpression";
  result.program = result.vars + " = myCast.stringExpression (" +
                   context->val->getText() + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitParentExpression(
    MyCParser::ParentExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes = std::any_cast<VisitRes>(visitExpression(context->expr));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.parentExpression";
  result.program = exprRes.program + result.vars + " = parentExpression (" +
                   exprRes.vars + " : " + exprRes.types + ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitStructureExpression(
    MyCParser::StructureExpressionContext *context) {
  VisitRes result;
  VisitRes valRes;
  for (auto *val : context->val)
    valRes += std::any_cast<VisitRes>(visitAssignmentExpression(val));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.parentExpression";
  result.program = valRes.program + result.vars + " = parentExpression (";
  if (!context->val.empty())
    result.program += valRes.vars + " : " + valRes.types;
  result.program += ")\n";
  return result;
}

std::any MyCFrontendVisitor::visitGenericExpression(
    MyCParser::GenericExpressionContext *context) {
  VisitRes result;
  VisitRes exprRes =
      std::any_cast<VisitRes>(visitAssignmentExpression(context->expr));
  VisitRes casesRes;
  for (auto *cs : context->cases)
    casesRes += std::any_cast<VisitRes>(visitGenericItem(cs));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.genericExpression";
  result.program = exprRes.program + casesRes.program + result.vars +
                   " = myCast.genericExpression ((" + exprRes.vars + " : " +
                   exprRes.types + "),(" + casesRes.vars + " : " +
                   casesRes.types + "))\n";
  return result;
}

std::any
MyCFrontendVisitor::visitGenericItem(MyCParser::GenericItemContext *context) {
  if (auto *ctx = dynamic_cast<MyCParser::TypeGenericItemContext *>(context))
    return visitTypeGenericItem(ctx);
  if (auto *ctx = dynamic_cast<MyCParser::DefaultGenericItemContext *>(context))
    return visitDefaultGenericItem(ctx);
  return VisitRes();
}

std::any MyCFrontendVisitor::visitTypeGenericItem(
    MyCParser::TypeGenericItemContext *context) {
  VisitRes result;
  VisitRes typeRes = std::any_cast<VisitRes>(visitType(context->type_));
  VisitRes bodyRes =
      std::any_cast<VisitRes>(visitAssignmentExpression(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.typeGenericItem";
  result.program = typeRes.program + bodyRes.program + result.vars +
                   " = myCast.typeGenericItem ((" + typeRes.vars + " : " +
                   typeRes.types + "),(" + bodyRes.vars + " : " +
                   bodyRes.types + "))\n";
  return result;
}

std::any MyCFrontendVisitor::visitDefaultGenericItem(
    MyCParser::DefaultGenericItemContext *context) {
  VisitRes result;
  VisitRes bodyRes =
      std::any_cast<VisitRes>(visitAssignmentExpression(context->body));
  result.vars = "%" + std::to_string(getNextVarIndex());
  result.types = "!myCast.defaultGenericItem";
  result.program = bodyRes.program + result.vars +
                   " = myCast.defaultGenericItem (" + bodyRes.vars + " : " +
                   bodyRes.types + ")\n";
  return result;
}