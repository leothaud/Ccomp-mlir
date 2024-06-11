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

class MyCast_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MyCast_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def MyCastAttrPred : CPred<[{
  return mlir::isa<MyCast_IntTypeAttr>($_self) ||
         mlir::isa<MyCast_FloatTypeAttr>($_self) ||
         mlir::isa<MyCast_PtrTyprAttr>($_self);
}]>;



def MyCast_IntTypeAttr : MyCast_Attr<"Integer", "int"> {
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "bool":$isSigned,
                        OptionalParameter<"uint64_t">:$bw);
                        
  let assemblyFormat = "`<` $isSigned (`,` $bw^)? `>`";
}

def MyCast_FloatTypeAttr : MyCast_Attr<"Float", "float"> {
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "bool":$isSigned,
                        OptionalParameter<"uint64_t">:$ebw,
                        OptionalParameter<"uint64_t">:$fbw);

  let assemblyFormat = "`<` $isSigned (`,` $ebw^ `,` $fbw)? `>`";
}

def MyCast_PtrTypeAttr : MyCast_Attr<"Ptr", "ptr"> {
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::Attribute":$baseType,
                        OptionalParameter<"uint64_t">:$size);

  let assemblyFormat = "`<` $baseType (`,` $size^)? `>`";

//  let genVerifyDecl = 1;
//  let verifier = [{
//    if (failed(MyCastAttrPred()($baseType)))
//      return emitOpError() << "PtrType base type must be a MyCast attribute.\n";
//    return success();
//  }];
}



#endif // MYCAST_ATTR