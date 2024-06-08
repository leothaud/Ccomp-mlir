//===--------------------------MyCastInterface.td--------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

#ifndef MYCAST_INTERFACE
#define MYCAST_INTERFACE

include "mlir/IR/OpBase.td"

def PrettyPrintOpInterface : OpInterface<"PrettyPrintOpInterface"> {
  let methods = [
    InterfaceMethod<
      "todo",
      "void", "prettyPrint", (ins "::std::ostringstream &":$out, "int":$indent), [{}], [{ {} }]
    >
  ];
  let cppNamespace = "::ccomp::myCast";
}

#endif // MYCAST_INTERFACE