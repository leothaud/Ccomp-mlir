#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Transforms/MyCast/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mlir/Support/LogicalResult.h>

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