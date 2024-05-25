get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        CastDialect
        )
add_llvm_executable(ccomp-opt ccomp-opt.cpp)

llvm_update_compile_flags(ccomp-opt)
target_link_libraries(ccomp-opt PRIVATE ${LIBS})