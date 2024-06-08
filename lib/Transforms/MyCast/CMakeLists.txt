add_mlir_library(MyCastTransforms
    PrettyPrintAst/prettyPrintAst.cpp
    CheckDefinedVar/checkDefinedVar.cpp

DEPENDS
    MyCastTransformsPassIncGen
    MLIRMyCastOpsIncGen

LINK_LIBS PUBLIC
    MyCastDialect
)