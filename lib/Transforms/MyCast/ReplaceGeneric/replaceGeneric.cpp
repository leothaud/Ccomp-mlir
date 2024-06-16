#include "Dialect/MyCast/MyCastAttr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Rewrite/PatternApplicator.h"

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <optional>

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace ccomp {
namespace myCast {

class ReplaceGenericPattern : public mlir::RewritePattern {
 public:
   ReplaceGenericPattern(mlir::MLIRContext *context) :
   mlir::RewritePattern(ccomp::myCast::GenericExpressionOp::getOperationName(),1,context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
     auto genOp = llvm::cast<GenericExpressionOp>(op);
     auto exprType = llvm::cast<MyCastInterface>(genOp.getExpr().getDefiningOp()).getTypeAttr();
     std::optional<mlir::OpResult> defaultExpr;
     for (auto case_: genOp.getCases()) {
      if (auto typeCase = llvm::dyn_cast<TypeGenericItemOp>(case_.getDefiningOp())) {
        if (areEquals(llvm::cast<MyCastInterface>(typeCase.getCond().getDefiningOp()).getTypeAttr(), exprType)) {
          rewriter.replaceAllUsesWith(genOp, typeCase.getBody().getDefiningOp()->getResult(0));
          rewriter.eraseOp(genOp);
          return mlir::success();
        }
      } else if (auto defaultCase = llvm::dyn_cast<DefaultGenericItemOp>(case_.getDefiningOp())) {
        defaultExpr = defaultCase.getBody().getDefiningOp()->getResult(0);
      }
     }
     rewriter.replaceAllUsesWith(genOp, *defaultExpr);
     rewriter.eraseOp(genOp);
     return mlir::success();
   }
};

template <class T>
mlir::LogicalResult applyPattern(mlir::Operation *op) {
  mlir::RewritePatternSet patterns(op->getContext());
  patterns.add<T>(op->getContext());
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  return mlir::applyPatternsAndFoldGreedily(op->getParentOp(),
                                            std::move(patterns), config);
}

struct ReplaceGenericPass : public impl::ReplaceGenericPassBase<ReplaceGenericPass> {
public:
  void runOnOperation() override {
    (void)applyPattern<ReplaceGenericPattern>(getOperation());
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createReplaceGenericPass() {
  return std::make_unique<ccomp::myCast::ReplaceGenericPass>();
}

} // namespace myCast
} // namespace ccomp