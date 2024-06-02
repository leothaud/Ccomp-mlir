#include "MyCFrontendVisitor.h"
#include "MyCParser.h"
#include "MyCLexer.h"
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " [MyC file] [output file]." << std::endl;
    return 1;
  }

  std::ifstream inStream(argv[1]);
  if (!inStream.is_open())
  {
    std::cerr << "Can't open file " << argv[1] << ": " << strerror(errno);
    exit(1);
  }
  std::ofstream outStream(argv[2]);
  if (!outStream.is_open())
  {
    std::cerr << "Can't open file " << argv[2] << ": " << strerror(errno);
    exit(1);
  }


  antlr4::ANTLRInputStream input(inStream);
  MyCLexer lexer(&input);
  
  antlr4::CommonTokenStream tokens(&lexer);
  MyCParser parser(&tokens);

  MyCParser::ProgramContext* program = parser.program();
  MyCFrontendVisitor visitor;

  VisitRes res = std::any_cast<VisitRes>(visitor.visitProgram(program));

  outStream << res.program;
  inStream.close();
  outStream.close();

  return 0;
}