#include "MyCFrontendVisitor.h"
#include "MyCParser.h"
#include "MyCLexer.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " [MyC file]." << std::endl;
    return 1;
  }

  std::ifstream stream(argv[1]);
  if (!stream.is_open())
  {
    std::cerr << "Can't open file " << argv[1] << ": " << strerror(errno);
    exit(1);
  }

  antlr4::ANTLRInputStream input(stream);
  MyCLexer lexer(&input);
  
  antlr4::CommonTokenStream tokens(&lexer);
  MyCParser parser(&tokens);

  MyCParser::ProgramContext* program = parser.program();
  MyCFrontendVisitor visitor;

  VisitRes res = std::any_cast<VisitRes>(visitor.visitProgram(program));

  std::cout << "vars = " << res.vars << std::endl
            << "types = " << res.types << std::endl
            << "program = {" << std::endl << res.program
            << std::endl << "}" << std::endl;

  return 0;
}