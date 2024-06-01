set(ANTLR4_JAR_LOCATION "${PROJECT_SOURCE_DIR}/thirdparty/antlr-complete.jar")
antlr4_generate(MyCAntlr ${PROJECT_SOURCE_DIR}/tools/frontend/MyC.g4 BOTH FALSE TRUE)
include_directories( ${PROJECT_BINARY_DIR} ${ANTLR4_INCLUDE_DIR} ${ANTLR4_INCLUDE_DIR_antlrcpptest_lexer} ${ANTLR4_INCLUDE_DIR_antlrcpptest_parser} )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable( Parsertest MyCFrontendMain.cpp MyCFrontendVisitor.cpp ${ANTLR4_SRC_FILES_MyCAntlr})
target_include_directories(Parsertest PRIVATE ${ANTLR4_INCLUDE_DIR_MyCAntlr})
add_dependencies( Parsertest antlr4_static )
target_link_libraries( Parsertest PRIVATE
                       antlr4_static)
install(TARGETS Parsertest DESTINATION bin)