//===--------------------------MyCcdfgOpsTypes.td--------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCCDFG_TYPES_TD__
#define MYCCDFG_TYPES_TD__

include "Dialect/MyCcdfg/MyCcdfgDialect.td"
include "mlir/IR/AttrTypeBase.td"

class MyCcdfg_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<MyCcdfg_Dialect, name, traits>
{
    let mnemonic = typeMnemonic;
}

def MyCcdfg_FooType : MyCcdfg_Type<"MyCcdfg_foo", "foo">{}

#endif // MYCCDFG_TYPES_TD__