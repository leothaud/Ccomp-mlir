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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

llvm::StringMap<mlir::Value> constVals;

class RetrieveConstPattern : public mlir::OpRewritePattern<BaseVarDeclOp> {
public:
  RetrieveConstPattern(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(BaseVarDeclOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool isConst = false;
    auto uses = op.getOperation()->getUses();
    bool first = true;
    for (auto &use : uses) {
      if (!first) {
        llvm::errs()
            << "Error, BaseVarDeclOp should only have one use for now.\n";
        op->dump();
        exit(1);
      }
      first = false;
      isConst = llvm::cast<MyCastInterface>(use.getOwner()).isConst();
    }
    if (isConst) {
      if (!op.getValue()) {
        llvm::errs() << "Error: const variable has no initial value.\n";
        op->dump();
        exit(1);
      }
      std::string varName = op.getName().getValue().data();
      constVals[varName] = op.getValue();
    }
    return mlir::failure();
  }
};

class ReplaceConstPattern : public mlir::OpRewritePattern<VarExpressionOp> {
public:
  ReplaceConstPattern(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(VarExpressionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    std::string varName = op.getName().getValue().data();
    if (constVals.contains(varName)) {
      rewriter.replaceAllUsesWith(op, constVals[varName]);
      return mlir::success();
    }
    return mlir::failure();
  }
};

class DeleteConstPattern : public mlir::OpRewritePattern<VarDeclOp> {
public:
  DeleteConstPattern(mlir::MLIRContext *context) : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(VarDeclOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.isConst()) {
      for (auto &use: op->getUses()) {
        if (auto stmt = llvm::dyn_cast<VarDeclStatementOp>(use.getOwner())) {
          rewriter.replaceAllUsesWith(stmt.getRes(), rewriter.create<CompoundStatementOp>(stmt.getLoc(), mlir::ValueRange()));
        } else {
          eraseOpFromProgram(op, rewriter);
        }
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct PropagateConstPass
    : public impl::PropagateConstPassBase<PropagateConstPass> {
public:
  void runOnOperation() override {

    auto *op = getOperation()->getParentOp();

    (void)applyPattern<RetrieveConstPattern>(op);
    (void)applyPattern<ReplaceConstPattern>(op);
    (void)applyPattern<DeleteConstPattern>(op);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createPropagateConstPass() {
  return std::make_unique<ccomp::myCast::PropagateConstPass>();
}

} // namespace myCast
} // namespace ccomp