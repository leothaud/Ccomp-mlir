//===------------------------ccomp-compiler.cpp---------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#include "Dialect/MyCast/MyCastDialect.h"
#include "Dialect/MyCast/MyCastOps.h"
#include "Dialect/MyCast/MyCastOpsDialect.cpp.inc"
#include "Dialect/MyCcdfg/MyCcdfgDialect.h"
#include "Dialect/MyCcdfg/MyCcdfgOpsDialect.cpp.inc"
#include "Dialect/initAllDialect.h"
#include "Transforms/MyCast/Passes.h"
#include "Transforms/initAllPasses.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "../frontend/MyCFrontendVisitor.h"
#include "MyCLexer.h"
#include "MyCParser.h"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <exception>
#include <iostream>

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "mlir/Parser/Parser.h"
#include "llvm/Support/InitLLVM.h"

#include "mlir/CAPI/Utils.h"

#include <filesystem>

struct Options {
  std::string inputFilename;

  bool outputToFile = false;
  std::string outputFilename;
};

void initOptions(Options &options, boost::program_options::variables_map &vm,
                 boost::program_options::options_description &description) {
  if (vm.count("help")) {
    std::cout << description << "\n";
    exit(0);
  }
  if (vm.count("input") == 0) {
    std::cerr << "Usage : ./ccomp-compiler [options] [input file]."
              << "\n";
    exit(1);
  }
  options.inputFilename = vm["input"].as<std::string>();

  if (vm.count("output")) {
    options.outputToFile = true;
    options.outputFilename = vm["output"].as<std::string>();
  } else {
    options.outputFilename =
        std::string(std::filesystem::temp_directory_path()) +
        std::string("/ccomp_res.tmp");
  }
}

int main(int argc, char **argv) {

  boost::program_options::options_description description("Compiler options");
  description.add_options()("input,i",
                            boost::program_options::value<std::string>(),
                            "input file")("help,h", "produce help message")(
      "output,o", boost::program_options::value<std::string>(), "output file")(
      "emit-mlir", "emit mlir file")("no-pass", "do not apply any pass");
  boost::program_options::variables_map vm;
  boost::program_options::positional_options_description p;
  p.add("input", 1);
  try {
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv)
            .options(description)
            .positional(p)
            .run(),
        vm);
    boost::program_options::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Usage : ./ccomp-compiler [options] [input file].\n"
              << "-------------------------------------------------------------"
                 "-------------------\n"
              << "Error:\n"
              << e.what() << '\n';
    exit(1);
  }

  Options options;
  initOptions(options, vm, description);

  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  ccomp::registerAllDialects(registry);
  ccomp::registerAllPasses();
  mlir::tracing::DebugConfig::registerCLOptions();

  auto inputStream = std::ifstream(options.inputFilename);

  antlr4::ANTLRInputStream input(inputStream);
  MyCLexer lexer(&input);

  antlr4::CommonTokenStream tokens(&lexer);
  MyCParser parser(&tokens);

  MyCParser::ProgramContext *program = parser.program();
  MyCFrontendVisitor visitor;

  std::string mlirText =
      std::any_cast<VisitRes>(visitor.visitProgram(program)).program;

  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);

  mlir::StringRef mlirRef = mlir::StringRef(mlirText);

  auto *moduleOp = mlir::parseSourceString<mlir::ModuleOp>(mlirRef, &context)
                       .release()
                       .getOperation();
  ccomp::myCast::ProgramOp programOp;
  for (auto &region : moduleOp->getRegions())
    for (auto &op : region.getOps())
      if (ccomp::myCast::ProgramOp prog =
              llvm::dyn_cast<ccomp::myCast::ProgramOp>(op)) {
        programOp = prog;
        goto ProgramOpFound;
      }
  llvm::errs() << "Error during parsing: no ccomp::myCast::ProgramOp found\n";
  exit(1);
ProgramOpFound:

  mlir::PassManager pm(&context);
  if (vm.count("no-pass") == 0) {
    pm.addPass(ccomp::myCast::createCheckDefinedVarPass());
    pm.addPass(ccomp::myCast::createCheckDeclaredFunPass());
    pm.addPass(ccomp::myCast::createReplaceAliasPass());
    pm.addPass(ccomp::myCast::createCompleteTypeAstPass());
    pm.addPass(ccomp::myCast::createReplaceGenericPass());
    pm.addPass(ccomp::myCast::createVariableUniquerPass());
    pm.addPass(ccomp::myCast::createPropagateConstPass());
    pm.addPass(ccomp::myCast::createForAndDoToWhilePass());
    pm.addPass(ccomp::myCast::createPullUpVarDeclsPass());
    pm.addPass(ccomp::myCast::createFlattenCompoundPass());
  }
  if (vm.count("emit-mlir")) {
    (void)pm.run(programOp);
    std::string outputString;
    llvm::raw_string_ostream os(outputString);
    programOp->getParentOp()->print(os);
    auto outputStream = std::ofstream(options.outputFilename);
    outputStream << outputString;
  } else {
    pm.addPass(ccomp::myCast::createPrettyPrintAstPass(options.outputFilename));
    (void)pm.run(programOp);
  }
  if (!options.outputToFile) {
    auto outputStream = std::ifstream(options.outputFilename);
    std::cout << outputStream.rdbuf();
  }

  return 0;
}