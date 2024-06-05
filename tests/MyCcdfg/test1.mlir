%0 = myCcdfg.intType
%1 = myCcdfg.scope (, [])
%2 = myCcdfg.funDef ((%0 : !myCcdfg.intType), "main", (%1 : !myCcdfg.scope), (%1 : !myCcdfg.scope)) {
    %3 = myCcdfg.unconditionalTransition ( %5 : !myCcdfg.basicBlock)
    %4 = myCcdfg.basicBlock (, (%3 : !myCcdfg.unconditionalTransition))
    %5 = myCcdfg.basicBlock(,)
}