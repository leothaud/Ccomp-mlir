#ifndef MYCUTILS_H__
#define MYCUTILS_H__

#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

template <class ...P>
static inline mlir::LogicalResult applyPattern(mlir::Operation *op, bool topDownTraversal = false) {
  mlir::RewritePatternSet patterns(op->getContext());
  ([&]{
    patterns.add<P>(op->getContext());
  } (), ...);
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = topDownTraversal;
  return mlir::applyPatternsAndFoldGreedily(op, std::move(patterns), config);
}


#endif // MYCUTILS_H__