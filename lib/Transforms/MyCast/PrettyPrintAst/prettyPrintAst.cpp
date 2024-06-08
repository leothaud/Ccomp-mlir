#include "mlir/IR/Builders.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOps.h.inc"
#include "Dialect/MyCast/MyCastOps.cpp.inc"
#include "Transforms/MyCast/Passes.h"
#include <fstream>
#include <memory>


using namespace mlir;

namespace ccomp {
    namespace myCast {
        


struct PrettyPrinterAstPass : public impl::PrettyPrintAstPassBase<PrettyPrinterAstPass> {
public:
  void runOnOperation() override {
    std::ofstream fileStream;
    fileStream.open(this->filename);
    if (!fileStream.is_open()) {
        llvm::errs() << "Error openning " << this->filename << '\n';
        exit(1);
    }
    std::ostringstream outputStream;
    auto programOp = this->getOperation();
    programOp.prettyPrint(outputStream, 0);
    fileStream << outputStream.str();
    fileStream.close();
  }
};

  std::unique_ptr<::mlir::OperationPass<ccomp::myCast::ProgramOp>> createPrettyPrintAstPass() {
    return std::make_unique<ccomp::myCast::PrettyPrinterAstPass>();
  }

    } // namespace myCast
} // namespace ccomp