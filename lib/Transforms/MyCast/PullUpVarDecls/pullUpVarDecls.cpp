#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/DialectConversion.h"
#include "myCutils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

class PullUpVarDeclsPattern : public mlir::RewritePattern {
public:
  mutable llvm::StringSet<> alreadySeen;
  PullUpVarDeclsPattern(mlir::MLIRContext *context)
      : RewritePattern(FunDefOp::getOperationName(), 1, context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *mlirOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto op = llvm::cast<FunDefOp>(mlirOp);
    if (!op.getBody())
      return mlir::failure();
    std::string funName = llvm::cast<FunProtoOp>(op.getProto().getDefiningOp())
                              .getName()
                              .getValue()
                              .data();
    if (alreadySeen.contains(funName))
      return mlir::failure();
    alreadySeen.insert(funName);
    auto *ctx = getContext();
    auto builder = mlir::Builder(ctx);
    llvm::SmallVector<mlir::Operation *> varDecls;
    op.getVarDecls(varDecls);
    llvm::SmallVector<mlir::Value> newDecls;
    for (auto *decl : varDecls) {
      VarDeclStatementOp declOp = llvm::cast<VarDeclStatementOp>(decl);
      llvm::SmallVector<mlir::Value> newStatement;
      llvm::SmallVector<mlir::Value> baseDecls;
      for (auto baseDecl :
           llvm::cast<VarDeclOp>(declOp.getDecl().getDefiningOp()).getDecls())
        baseDecls.push_back(baseDecl);
      for (auto baseDecl : baseDecls) {
        auto baseDeclOp = llvm::cast<BaseVarDeclOp>(baseDecl.getDefiningOp());
        auto loc = baseDeclOp.getLoc();
        if (baseDeclOp.getValue()) {
          mlir::Value lval =
              rewriter
                  .create<VarExpressionOp>(
                      loc, builder.getStringAttr(
                               baseDeclOp.getName().getValue().data()))
                  .getRes();
          mlir::Value oper = rewriter.create<EqOpOp>(loc).getRes();
          mlir::Value rval =
              rewriter.clone(*baseDeclOp.getValue().getDefiningOp())
                  ->getResult(0);
          mlir::Value affec =
              rewriter.create<AssignmentExpressionOp>(loc, lval, oper, rval)
                  .getRes();
          mlir::Value stmt =
              rewriter.create<ExpressionStatementOp>(loc, affec).getRes();

          newStatement.push_back(stmt);
          baseDeclOp.getValueMutable().clear();
        }
      }
      auto compound =
          rewriter.create<CompoundStatementOp>(decl->getLoc(), newStatement);
      rewriter.replaceAllUsesWith(decl->getResult(0), compound.getRes());
      newDecls.push_back(declOp.getRes());
    }
    newDecls.push_back(
        rewriter.clone(*op.getBody().getDefiningOp())->getResult(0));
    auto compound = rewriter
                        .create<CompoundStatementOp>(op.getBody().getLoc(),
                                                     mlir::ValueRange(newDecls))
                        .getRes();
    rewriter.replaceAllUsesWith(op.getBody().getDefiningOp()->getResult(0),
                                compound);
    return mlir::success();
  }
};

struct PullUpVarDeclsPass
    : public impl::PullUpVarDeclsPassBase<PullUpVarDeclsPass> {
public:
  void runOnOperation() override {

    auto *op = getOperation()->getParentOp();
    (void)applyPattern<PullUpVarDeclsPattern>(op);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createPullUpVarDeclsPass() {
  return std::make_unique<ccomp::myCast::PullUpVarDeclsPass>();
}

} // namespace myCast
} // namespace ccomp