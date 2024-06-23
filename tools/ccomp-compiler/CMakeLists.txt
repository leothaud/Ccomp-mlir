get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MyCastTransforms
        MyCastDialect
        MyCcdfgDialect
        )
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
set(ANTLR4_JAR_LOCATION "${PROJECT_SOURCE_DIR}/thirdparty/antlr-complete.jar")
antlr4_generate(MyCAntlr ${PROJECT_SOURCE_DIR}/tools/ccomp-compiler/MyC.g4 BOTH FALSE TRUE)
include_directories( ${PROJECT_BINARY_DIR} ${ANTLR4_INCLUDE_DIR} ${ANTLR4_INCLUDE_DIR_antlrcpptest_lexer} ${ANTLR4_INCLUDE_DIR_antlrcpptest_parser} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(ccomp-compiler ccomp-compiler.cpp MyCFrontendVisitor.cpp ${ANTLR4_SRC_FILES_MyCAntlr})


TARGET_LINK_LIBRARIES(ccomp-compiler LINK_PUBLIC ${Boost_LIBRARIES} )

target_include_directories(ccomp-compiler PRIVATE ${ANTLR4_INCLUDE_DIR_MyCAntlr})
add_dependencies( ccomp-compiler antlr4_static )
target_link_libraries(ccomp-compiler PRIVATE
                        antlr4_static)


llvm_update_compile_flags(ccomp-compiler)
target_link_libraries(ccomp-compiler PRIVATE ${LIBS})
install(TARGETS ccomp-compiler DESTINATION bin)