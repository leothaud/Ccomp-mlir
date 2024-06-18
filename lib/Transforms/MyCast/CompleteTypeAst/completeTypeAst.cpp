#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

struct CompleteTypeAstPass
    : public impl::CompleteTypeAstPassBase<CompleteTypeAstPass> {
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