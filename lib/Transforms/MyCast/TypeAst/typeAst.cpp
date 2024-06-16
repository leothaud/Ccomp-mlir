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
#include "mlir/Transforms/DialectConversion.h"

namespace ccomp {
namespace myCast {

struct TypeAstPass : public impl::TypeAstPassBase<TypeAstPass> {
public:
  void runOnOperation() override {

    llvm::StringMap<mlir::Attribute> globals;
    llvm::StringMap<mlir::Attribute> locals;
    FunctionMap funMap;
    llvm::StringMap<llvm::StringMap<mlir::Attribute>> fieldMap;
    ProgramOp op = getOperation();
    (void)op.addType(globals, locals, funMap, fieldMap);
  }
};

std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>>
createTypeAstPass() {
  return std::make_unique<ccomp::myCast::TypeAstPass>();
}

} // namespace myCast
} // namespace ccomp