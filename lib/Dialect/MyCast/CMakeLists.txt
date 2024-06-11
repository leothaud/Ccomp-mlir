add_mlir_dialect_library(MyCastDialect
        MyCastDialect.cpp
        MyCastOps.cpp
        MyCastOpsTypes.cpp
        MyCastAttr.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/MyCast

        DEPENDS
        MLIRMyCastOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)