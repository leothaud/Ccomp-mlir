set(ANTLR4_JAR_LOCATION "${PROJECT_SOURCE_DIR}/thirdparty/antlr-complete.jar")
antlr4_generate(MyCAntlr ${PROJECT_SOURCE_DIR}/tools/frontend/MyC.g4 BOTH FALSE TRUE)
include_directories( ${PROJECT_BINARY_DIR} ${ANTLR4_INCLUDE_DIR} ${ANTLR4_INCLUDE_DIR_antlrcpptest_lexer} ${ANTLR4_INCLUDE_DIR_antlrcpptest_parser} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable( ccomp-parser MyCFrontendMain.cpp MyCFrontendVisitor.cpp ${ANTLR4_SRC_FILES_MyCAntlr})
target_include_directories(ccomp-parser PRIVATE ${ANTLR4_INCLUDE_DIR_MyCAntlr})
add_dependencies( ccomp-parser antlr4_static )
target_link_libraries( ccomp-parser PRIVATE
                       antlr4_static)
install(TARGETS ccomp-parser DESTINATION bin)