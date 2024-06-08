add_mlir_library(MyCastTransforms
    PrettyPrintAst/prettyPrintAst.cpp

DEPENDS
    MyCastTransformsPassIncGen
    MLIRMyCastOpsIncGen

LINK_LIBS PUBLIC
    MyCastDialect
)