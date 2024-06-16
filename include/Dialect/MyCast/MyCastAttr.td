#ifndef MYCAST_ATTR
#define MYCAST_ATTR

include "Dialect/MyCast/MyCastDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


class MyCast_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MyCast_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def MyCast_IntTypeAttr : MyCast_Attr<"Integer", "int"> {
  let parameters = (ins "bool":$isSigned,
                        OptionalParameter<"uint64_t">:$bw);
                        
  let assemblyFormat = "`<` $isSigned (`,` $bw^)? `>`";
}

def MyCast_FloatTypeAttr : MyCast_Attr<"Float", "float"> {
  let parameters = (ins "bool":$isSigned,
                        OptionalParameter<"uint64_t">:$ebw,
                        OptionalParameter<"uint64_t">:$mbw);

  let assemblyFormat = "`<` $isSigned (`,` $ebw^ `,` $mbw)? `>`";
}

def MyCast_PtrTypeAttr : MyCast_Attr<"Ptr", "ptr"> {
  let parameters = (ins "mlir::Attribute":$baseType,
                        OptionalParameter<"uint64_t">:$size);

  let assemblyFormat = "`<` $baseType (`,` $size^)? `>`";

  let genVerifyDecl = 1;

  code verifier = [{
  mlir::LogicalResult ccomp::myCast::PtrAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic ()> emitError, mlir::Attribute baseType, unsigned long size) {
    if (mlir::isa<IntegerAttr>(baseType) ||
         mlir::isa<FloatAttr>(baseType) ||
         mlir::isa<PtrAttr>(baseType) ||
         mlir::isa<EnumAttr>(baseType) ||
         mlir::isa<UnionAttr>(baseType) ||
         mlir::isa<StructAttr>(baseType) ||
         mlir::isa<VoidAttr>(baseType) ||
         mlir::isa<UntypedAttr>(baseType)
         )
      return mlir::success();
    return emitError() << "Error: base type of pointer type must be MyCast type attribute.\n";
  }
  }];

  let extraClassDefinition = verifier # [{}];
}

def MyCast_EnumTypeAttr : MyCast_Attr<"Enum", "enum"> {
  let parameters = (ins Builtin_StringAttr:$name);
  let assemblyFormat = "`<` $name `>`";
}

def MyCast_StructTypeAttr : MyCast_Attr<"Struct", "struct"> {
  let parameters = (ins Builtin_StringAttr:$name);
  let assemblyFormat = "`<` $name `>`";
}

def MyCast_UnionTypeAttr : MyCast_Attr<"Union", "union"> {
  let parameters = (ins Builtin_StringAttr:$name);
  let assemblyFormat = "`<` $name `>`";
}

def MyCast_VoidTypeAttr : MyCast_Attr<"Void", "void"> {
  let parameters = (ins );
  let assemblyFormat = "";
}

def MyCast_UntypedAttr: MyCast_Attr<"Untyped", "untyped"> {
  let parameters = (ins );
  let assemblyFormat = "";
}

#endif // MYCAST_ATTR