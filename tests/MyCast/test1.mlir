//int main(void) {
//    return 0;
//}

%returnType = myCast.intType { x=#myCast.int<false>, y= #myCast.float<true,3,7>, z=#myCast.ptr<3>}
%returnValue = myCast.intExpression (0)
%returnValue2 = myCast.intExpression (0)
%return = myCast.returnStatement (%returnValue : !myCast.intExpression)
%mainBody = myCast.compoundStatement (%return : !myCast.returnStatement)
%intType = myCast.intType
%mainArgcArg = myCast.argument ((%intType : !myCast.intType), "argc", )
%charType = myCast.charType
%charPtrPtrType = myCast.type (, (%charType : !myCast.charType), 2)
%mainArgvArg = myCast.argument ((%intType : !myCast.intType), "argc", )
%mainProto = myCast.funProto ((%returnType : !myCast.intType), "main", (%mainArgcArg, %mainArgvArg : !myCast.argument, !myCast.argument))
%mainFun = myCast.funDef ((%mainProto : !myCast.funProto), (%mainBody : !myCast.compoundStatement))

%voidType = myCast.voidType
%definedProto = myCast.funProto ((%voidType : !myCast.voidType), "bloubli",)
%definedFun = myCast.funDef ((%definedProto : !myCast.funProto),)

%program = myCast.program (%mainFun, %definedFun : !myCast.funDef, !myCast.funDef)