//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CAST_TYPES_TD__
#define CAST_TYPES_TD__

include "Dialect/Cast/CastDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Cast_AstNodeType<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<Cast_Dialect, name, traits>
{
  let mnemonic = typeMnemonic;
}

#endif
