get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRAnalysis
        MLIRDialect
        MLIRFuncAllExtensions
        MLIRLspServerLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        MyCastDialect
)
add_llvm_executable(ccomp-lsp-server ccompLspServerMain.cpp)
llvm_update_compile_flags(ccomp-lsp-server)
target_link_libraries(ccomp-lsp-server PRIVATE ${LIBS})
install(TARGETS ccomp-lsp-server DESTINATION bin)