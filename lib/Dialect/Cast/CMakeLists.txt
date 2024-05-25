add_mlir_dialect_library(CastDialect
        CastDialect.cpp
        CastOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Cast

        DEPENDS
        MLIRCastOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)