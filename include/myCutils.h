#ifndef MYCUTILS_H__
#define MYCUTILS_H__

#include "Dialect/MyCast/MyCastOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

template <class... P>
static inline mlir::LogicalResult applyPattern(mlir::Operation *op,
                                               bool topDownTraversal = false) {
  mlir::RewritePatternSet patterns(op->getContext());
  ([&] { patterns.add<P>(op->getContext()); }(), ...);
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = topDownTraversal;
  return mlir::applyPatternsAndFoldGreedily(op, std::move(patterns), config);
}

static inline void eraseOpFromProgram(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) {
  for (auto &use : op->getUses()) {
    auto *owner = use.getOwner();

    rewriter.startOpModification(owner);
    auto prog = llvm::cast<ccomp::myCast::ProgramOp>(owner);
    auto items = prog.getItemsMutable();
    items.erase(use.getOperandNumber());
    rewriter.finalizeOpModification(owner);
  }
  rewriter.eraseOp(op);
}

#endif // MYCUTILS_H__