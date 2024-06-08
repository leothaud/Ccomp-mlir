#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>


namespace ccomp {
namespace myCast {

struct CheckDefinedVarPass
    : public impl::CheckDefinedVarPassBase<CheckDefinedVarPass> {
public:
  void runOnOperation() override {
    this->getOperation().checkDefinedVar(llvm::SmallVector<std::string>());
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createCheckDefinedVarPass() {
  return std::make_unique<ccomp::myCast::CheckDefinedVarPass>();
}

} // namespace myCast
} // namespace ccomp