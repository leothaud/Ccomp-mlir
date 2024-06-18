#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace ccomp {
namespace myCast {

struct CheckDeclaredFunPass
    : public impl::CheckDeclaredFunPassBase<CheckDeclaredFunPass> {
public:
  void runOnOperation() override {
    llvm::SmallVector<std::string> declared;
    this->getOperation().checkDeclaredFun(declared);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createCheckDeclaredFunPass() {
  return std::make_unique<ccomp::myCast::CheckDeclaredFunPass>();
}

} // namespace myCast
} // namespace ccomp