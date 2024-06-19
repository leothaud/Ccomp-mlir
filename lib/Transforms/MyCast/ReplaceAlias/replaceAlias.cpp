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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

class AliasRemoverPattern : public mlir::RewritePattern {
public:
  mutable llvm::StringMap<mlir::Operation *> aliases;

  AliasRemoverPattern(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (auto td = llvm::dyn_cast<ccomp::myCast::AliasDefOp>(op)) {
      std::string aliasName = td.getName().getValue().data();
      aliases[aliasName] = td.getBaseType().getDefiningOp();
      return mlir::success();
    }
    if (auto td = llvm::dyn_cast<ccomp::myCast::AliasTypeOp>(op)) {
      std::string typeName = td.getName().getValue().data();
      if (!aliases.contains(typeName)) {
        llvm::errs() << "Error: type " << typeName << " not found.\ntypes:\n";
        for (auto alias : aliases.keys())
          llvm::errs() << alias << "\n";
        exit(1);
      }
      auto newValue = aliases[typeName]->getResult(0);
      rewriter.replaceAllUsesWith(td, newValue);
      return mlir::success();
    }
    return mlir::failure();
  }
};

class TypedefRemoverPattern : public mlir::RewritePattern {
public:
  TypedefRemoverPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(AliasDefOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (auto td = llvm::dyn_cast<AliasDefOp>(op)) {
      eraseOpFromProgram(td, rewriter);

      return mlir::success();
    }
    return mlir::failure();
  }
};

struct ReplaceAliasPass : public impl::ReplaceAliasPassBase<ReplaceAliasPass> {
public:
  void runOnOperation() override {

    auto *op = getOperation()->getParentOp();
    (void)applyPattern<AliasRemoverPattern>(op, true);
    (void)applyPattern<TypedefRemoverPattern>(op, true);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createReplaceAliasPass() {
  return std::make_unique<ccomp::myCast::ReplaceAliasPass>();
}

} // namespace myCast
} // namespace ccomp