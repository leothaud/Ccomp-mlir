//int main(void) {
//    return 0;
//}

%returnType = myCast.intType
%returnValue = myCast.intExpression (0)
%returnValue2 = myCast.intExpression (0)
%return = myCast.returnStatement (%returnValue : !myCast.intExpression)
%mainBody = myCast.compoundStatement (%return : !myCast.returnStatement)
%mainProto = myCast.funProto ((%returnType : !myCast.intType), "main",)
%mainFun = myCast.funDef ((%mainProto : !myCast.funProto), (%mainBody : !myCast.compoundStatement))

%decl = myCast.baseVarDecl ( "x", (%returnValue, %returnValue2 : !myCast.intExpression, !myCast.intExpression) , (%returnValue2 : !myCast.intExpression))