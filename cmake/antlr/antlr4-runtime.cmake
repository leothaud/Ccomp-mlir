set(ANTLR_VERSION @ANTLR_VERSION@)

set(ANTLR4_INCLUDE_DIR "@PACKAGE_ANTLR4_INCLUDE_DIR@")
set(ANTLR4_LIB_DIR "@PACKAGE_ANTLR4_LIB_DIR@")

include(CMakeFindDependencyMacro)
find_dependency(Threads)

include(${CMAKE_CURRENT_LIST_DIR}/@targets_export_name@.cmake)

check_required_components(antlr)