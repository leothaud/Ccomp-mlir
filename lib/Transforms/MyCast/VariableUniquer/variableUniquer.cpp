#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

namespace ccomp {
namespace myCast {

struct VariableUniquerPass
    : public impl::VariableUniquerPassBase<VariableUniquerPass> {
public:
  void runOnOperation() override {
    llvm::StringMap<std::string> renamer;
    llvm::StringSet<> usedVar;
    auto op = getOperation();
    op.variableUniquer(renamer, usedVar);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createVariableUniquerPass() {
  return std::make_unique<ccomp::myCast::VariableUniquerPass>();
}

} // namespace myCast
} // namespace ccomp