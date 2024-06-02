add_mlir_dialect_library(MyCcdfgDialect
        MyCcdfgDialect.cpp
        MyCcdfgOps.cpp
        MyCcdfgOpsTypes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MyCcdfg

        DEPENDS
        MLIRMyCcdfgOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)