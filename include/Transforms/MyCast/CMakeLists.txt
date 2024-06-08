set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(MyCastTransformsPassIncGen)
add_mlir_doc(Passes MYCASTGeneralPasses ./ -gen-pass-doc)