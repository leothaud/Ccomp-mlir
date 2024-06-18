#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/DialectConversion.h"
#include "myCutils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

class CompoundInlinerPattern
    : public mlir::OpRewritePattern<CompoundStatementOp> {
public:
  CompoundInlinerPattern(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(CompoundStatementOp op,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> stmts;
    for (auto stmt : op.getStmt()) {
      if (auto compound = llvm::dyn_cast_if_present<CompoundStatementOp>(
              stmt.getDefiningOp())) {
        for (auto val : compound.getStmt())
          stmts.push_back(val);
      } else {
        stmts.push_back(stmt);
      }
    }
    auto mutableStmt = op.getStmtMutable();
    mutableStmt.clear();
    for (auto val : stmts)
      mutableStmt.append(val);
    return mlir::success();
  }
};

struct InlineCompoundPass
    : public impl::InlineCompoundPassBase<InlineCompoundPass> {
public:
  void runOnOperation() override {
    (void)applyPattern<CompoundInlinerPattern>(getOperation()->getParentOp());
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createInlineCompoundPass() {
  return std::make_unique<ccomp::myCast::InlineCompoundPass>();
}

} // namespace myCast
} // namespace ccomp