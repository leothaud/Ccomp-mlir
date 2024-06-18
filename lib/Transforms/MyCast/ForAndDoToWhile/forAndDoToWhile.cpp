#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Dialect/MyCast/MyCastOpsTypes.h"
#include "Transforms/MyCast/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/DialectConversion.h"
#include "myCutils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

class ForToWhilePattern : public mlir::OpRewritePattern<ForStatementOp> {
public:
  mutable llvm::StringMap<mlir::Operation *> aliases;

  ForToWhilePattern(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(ForStatementOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto *ctx = getContext();
    auto builder = mlir::Builder(ctx);
    llvm::SmallVector<mlir::Value> compoundValues;
    llvm::SmallVector<mlir::Value> whileValues;

    if (op.getInit()) {
      for (auto val :
           rewriter.create<VarDeclStatementOp>(op.getLoc(), op.getInit())
               ->getResults())
        compoundValues.push_back(val);
    }

    whileValues.push_back(op.getBody());
    if (op.getStep()) {
      auto stepStmt =
          rewriter.create<ExpressionStatementOp>(op.getLoc(), op.getStep());
      for (auto val : stepStmt->getResults())
        whileValues.push_back(val);
    }
    auto whileCond = op.getCond()
                         ? op.getCond()
                         : rewriter.create<IntExpressionOp>(
                               op.getLoc(), MyCast_IntExpressionType::get(ctx),
                               builder.getI32IntegerAttr(1));

    mlir::ValueRange range2(whileValues);
    auto whileBody = rewriter.create<CompoundStatementOp>(
        op.getLoc(), MyCast_CompoundStatementType::get(ctx), range2);
    auto whileStmt = rewriter.create<WhileStatementOp>(
        op.getLoc(), MyCast_WhileStatementType::get(ctx), whileCond, whileBody);

    for (auto val : whileStmt->getResults())
      compoundValues.push_back(val);

    mlir::ValueRange range(compoundValues);
    auto compound = rewriter.create<CompoundStatementOp>(
        op.getLoc(), MyCast_CompoundStatementType::get(ctx), range);

    rewriter.replaceOp(op, compound);
    return mlir::success();
  }
};

class DoToWhilePattern : public mlir::OpRewritePattern<DoWhileStatementOp> {
public:
  mutable llvm::StringMap<mlir::Operation *> aliases;

  DoToWhilePattern(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(DoWhileStatementOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto *ctx = getContext();
    mlir::ValueRange range, range2;
    auto compound = rewriter.create<CompoundStatementOp>(
        op.getLoc(), MyCast_CompoundStatementType::get(ctx), range);

    compound.getStmtMutable().append(
        rewriter.clone(*op.getBody().getDefiningOp())->getResults());
    auto whileStmt = rewriter.create<WhileStatementOp>(
        op.getLoc(), MyCast_WhileStatementType::get(ctx), op.getCond(),
        op.getBody());

    compound.getStmtMutable().append(whileStmt->getResults());
    rewriter.replaceOp(op, compound);
    return mlir::success();
  }
};

struct ForAndDoToWhilePass
    : public impl::ForAndDoToWhilePassBase<ForAndDoToWhilePass> {
public:
  void runOnOperation() override {
    (void)applyPattern<ForToWhilePattern, DoToWhilePattern>(
        getOperation()->getParentOp(), true);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createForAndDoToWhilePass() {
  return std::make_unique<ccomp::myCast::ForAndDoToWhilePass>();
}

} // namespace myCast
} // namespace ccomp