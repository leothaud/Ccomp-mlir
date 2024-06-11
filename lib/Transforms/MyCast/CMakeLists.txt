add_mlir_library(MyCastTransforms
    PrettyPrintAst/prettyPrintAst.cpp
    CheckDefinedVar/checkDefinedVar.cpp
    CheckDeclaredFun/checkDeclaredFun.cpp
    ReplaceAlias/replaceAlias.cpp

DEPENDS
    MyCastTransformsPassIncGen
    MLIRMyCastOpsIncGen

LINK_LIBS PUBLIC
    MyCastDialect
)