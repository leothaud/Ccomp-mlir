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
#include <optional>

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"


namespace ccomp {
namespace myCast {

struct CompleteTypeAstPass : public impl::CompleteTypeAstPassBase<CompleteTypeAstPass> {
public:
  void runOnOperation() override {
    mlir::PassManager pm(&getContext());
    pm.addPass(createTypeAstPass());
    pm.addPass(createReplaceGenericPass());
    pm.addPass(mlir::createCSEPass());
    (void)pm.run(getOperation());
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createCompleteTypeAstPass() {
  return std::make_unique<ccomp::myCast::CompleteTypeAstPass>();
}

} // namespace myCast
} // namespace ccomp