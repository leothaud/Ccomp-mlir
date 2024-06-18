add_mlir_library(MyCastTransforms
    PrettyPrintAst/prettyPrintAst.cpp
    CheckDefinedVar/checkDefinedVar.cpp
    CheckDeclaredFun/checkDeclaredFun.cpp
    ReplaceAlias/replaceAlias.cpp
    TypeAst/typeAst.cpp
    ReplaceGeneric/replaceGeneric.cpp
    CompleteTypeAst/completeTypeAst.cpp
    VariableUniquer/variableUniquer.cpp
    ForAndDoToWhile/forAndDoToWhile.cpp
    FlattenCompound/flattenCompound.cpp

DEPENDS
    MyCastTransformsPassIncGen
    MLIRMyCastOpsIncGen

LINK_LIBS PUBLIC
    MyCastDialect
)