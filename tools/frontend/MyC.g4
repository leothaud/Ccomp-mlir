//===--------------------------MyC.g4--------------------------------------===//
//
// Part of the Ccomp project.
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------- Copyright 2024 Dylan Leothaud --------------------------===//

grammar MyC;

WHITESPACE: [ \t\r\n] -> channel(HIDDEN);
LINECOMMENT: '//' ~[\n]* -> channel(HIDDEN);
BLOCKCOMMENT: '/*' .*? '*/' -> channel(HIDDEN);

LEFTPARENT: '(';
RIGHTPARENT: ')';
LEFTBRACKET: '{';
RIGHTBRACKET: '}';
LEFTSBRACKET: '[';
RIGHTSBRACKET: ']';

COMMA: ',';
SEMI: ';';
COLON: ':';

DOUBLEPLUS: '++';
DOUBLEMINUS: '--';
PLUSEQ: '+=';
MINUSEQ: '-=';
DOUBLEEQ: '==';
NOTEQ: '!=';
STAREQ: '*=';
DIVEQ: '/=';
LOR:'||';
OR:'|';
LSHIFTEQ: '<<=';
RSHIFTEQ: '>>=';
LSHIFT: '<<';
RSHIFT: '>>';
LE: '<=';
LT: '<';
GE: '>=';
GT: '>';
ANDEQ: '&=';
OREQ: '|=';
XOREQ: '^=';
XOR: '^';
LAND:'&&';
AND:'&';
MODULOEQ: '%=';
MODULO:'%';
DIV:'/';
PLUS: '+';
MINUS: '-';
STAR: '*';
EQUAL: '=';
QUESTIONMARK: '?';
LNOT: '!';
NOT: '~';
DOT: '.';
TO: '->';

GENERIC: '_Generic';
SIZEOF: 'sizeof';
INLINE: 'inline';
ENUM: 'enum';
UNION: 'union';
STRUCT: 'struct';
IF: 'if';
ELSE: 'else';
SWITCH: 'switch';
CASE: 'case';
DEFAULT: 'default';
WHILE: 'while';
FOR: 'for';
DO: 'do';
GOTO: 'goto';
BREAK: 'break';
CONTINUE: 'continue';
RETURN: 'return';
EXTERN: 'extern';
STATIC: 'static';
REGISTER: 'register';
CONST: 'const';
VOLATILE: 'volatile';
VOID: 'void';
CHAR: 'char';
SHORT: 'short';
UNSIGNED: 'unsigned';
INT: 'int';
LONG: 'long';
FLOAT: 'float';
DOUBLE: 'double';
TYPEDEF: 'typedef';

ID: [a-zA-Z] [a-zA-Z0-9_]*;
FLOATCST: [0-9]+ '.' [0-9]+;
INTCST: [0-9]+;
STRINGCST: '"' ~["] '"';

program:
    items+=programItem+;

programItem:
    funDef
    | varDecl SEMI
    | typeDef SEMI
;

funDef: proto=funProto (SEMI | body+=compoundStatement);

funProto: returnType=type name=ID LEFTPARENT (args+=argument (COMMA args+=argument)*)? RIGHTPARENT;

argument: type_=type name=ID (LEFTSBRACKET sizes+=expression RIGHTSBRACKET)*;

varDecl: type_=type decls+=baseVarDecl (COMMA decls+=baseVarDecl)*;

baseVarDecl: name=ID (LEFTSBRACKET sizes+=expression? RIGHTSBRACKET)* (EQUAL value+=assignmentExpression)?;

typeDef:
    UNION name=ID? LEFTBRACKET (decls+=varDecl SEMI)+ RIGHTBRACKET #unionDef
    | STRUCT name=ID? LEFTBRACKET (decls+=varDecl SEMI)+ RIGHTBRACKET #structDef
    | ENUM name=ID? LEFTBRACKET items+=enumItem (COMMA items+=enumItem)* RIGHTBRACKET #enumDef
    | TYPEDEF bt=type name=ID #aliasDef
;
enumItem: name=ID (EQUAL value=conditionalExpression)?;

statement:
    labeledStatement
    | compoundStatement
    | varDeclStatement
    | expressionStatement
    | ifStatement
    | switchStatement
    | whileStatement
    | doWhileStatement
    | forStatement
    | gotoStatement
    | continueStatement
    | breakStatement
    | returnStatement
;

labeledStatement: label=ID COLON stmt=statement;
compoundStatement: LEFTBRACKET stmt+=statement* RIGHTBRACKET;
varDeclStatement: decl=varDecl SEMI;
expressionStatement: value=expression? SEMI;
ifStatement: IF LEFTPARENT cond=expression RIGHTPARENT thenPart=statement (ELSE elsePart=statement)?;
switchStatement:
    SWITCH LEFTPARENT cond=expression RIGHTPARENT
    ((items+=switchItem) |
    (LEFTBRACKET items+=switchItem* RIGHTBRACKET))
;
switchItem:
    CASE cond=expression COLON body=statement #switchCaseItem
    | DEFAULT COLON body=statement #switchDefaultItem;
whileStatement: WHILE LEFTPARENT cond=expression RIGHTPARENT body=statement;
doWhileStatement: DO body=statement WHILE LEFTPARENT cond=expression RIGHTPARENT SEMI;
forStatement: FOR LEFTPARENT init=varDecl? SEMI cond=expression? SEMI step=expression? RIGHTPARENT body=statement;
gotoStatement: GOTO label=ID SEMI;
continueStatement: CONTINUE SEMI;
breakStatement: BREAK SEMI;
returnStatement: RETURN value=expression? SEMI;

type: 
    child=baseType
    | modifiers+=typeModifier* bt=baseType stars+=STAR*;
typeModifier:
    STATIC #staticTypeModifier
    | CONST #constTypeModifier
    | EXTERN #externTypeModifier
    | VOLATILE #volatileTypeModifier
;

baseType:
    VOID #voidType
    | UNSIGNED LONG LONG #unsignedLongLongType
    | UNSIGNED LONG #unsignedLongType
    | UNSIGNED SHORT #unsignedShortType
    | UNSIGNED CHAR #unsignedCharType
    | UNSIGNED INT? #unsignedIntType
    | CHAR #charType
    | SHORT #shortType
    | INT #intType
    | LONG LONG #longLongType
    | LONG #longType
    | FLOAT #floatType
    | DOUBLE #doubleType
    | ENUM name=ID #enumType
    | STRUCT name=ID #structType
    | UNION name=ID #unionType
    | name=ID #aliasType
;

expression: assignmentExpression;
assignmentExpression:
    child=conditionalExpression
    | lvalue=unaryExpression op=assignOperator rvalue=assignmentExpression;
assignOperator:
    EQUAL #eqOp
    | STAREQ #starEqOp
    | DIVEQ #divEqOp
    | MODULOEQ #moduloEqOp
    | PLUSEQ #plusEqOp
    | MINUSEQ #minusEqOp
    | LSHIFTEQ #leftShiftEqOp
    | RSHIFTEQ #rightShiftEqOp
    | ANDEQ #andEqOp
    | XOREQ #xorEqOp
    | OREQ #orEqOp
;

conditionalExpression:
    child=logicalOrExpression
    | cond=logicalOrExpression QUESTIONMARK thenPart=expression COLON elsePart=conditionalExpression
;

logicalOrExpression:
    child=logicalAndExpression
    | expr+=logicalAndExpression (LOR expr+=logicalAndExpression)+
;

logicalAndExpression:
    child=orExpression
    | expr+=orExpression (LAND expr+=orExpression)+
;

orExpression:
    child=xorExpression
    | expr+=xorExpression (OR expr+=xorExpression)+
;

xorExpression:
    child=andExpression
    | expr+=andExpression (XOR expr+=andExpression)+
;

andExpression:
    child=equalityExpression
    | expr+=equalityExpression (AND expr+=equalityExpression)+
;

equalityExpression:
    child=relationalExpression
    | lval=equalityExpression op=equalityOperator rval=relationalExpression
;
equalityOperator:
    DOUBLEEQ #equalOperator
    | NOTEQ #notEqualOperator
;
relationalExpression:
    child=shiftExpression
    | lval=relationalExpression op=relationalOperator rval=shiftExpression
;
relationalOperator:
    GE #geOperator
    | GT #gtOperator
    | LE #leOperator
    | LT #ltOperator
;
shiftExpression:
    child=additiveExpression
    | lval=shiftExpression op=shiftOperator rval=additiveExpression
;
shiftOperator:
    LSHIFT #lshiftOperator
    | RSHIFT #rshiftOperator
;
additiveExpression:
    child=multiplicativeExpression
    | lval=additiveExpression op=additiveOperator rval=multiplicativeExpression
;
additiveOperator:
    PLUS #plusOperator
    | MINUS #minusOperator
;
multiplicativeExpression:
    child=castExpression
    | lval=multiplicativeExpression op=multiplicativeOperator rval=castExpression
;
multiplicativeOperator:
    STAR #multOperator
    | DIV #divOperator
    | MODULO #moduloOperator
;
castExpression:
    child=unaryExpression
    | LEFTPARENT newType=type RIGHTPARENT expr=unaryExpression;

unaryExpression:
    postfixExpression
    | sizeofExpression
    | unopExpression
;
unopExpression:op=unaryOperator expr=unaryExpression;
unaryOperator:
    DOUBLEPLUS #incrOperator
    | DOUBLEMINUS #decrOperator
    | AND #addrofOperator
    | STAR #derefOperator
    | PLUS #positiveOperator
    | MINUS #negativeOperator
    | NOT #notOperator
    | LNOT #lnotOperator
;
sizeofExpression: SIZEOF (expr=unaryExpression | (LEFTPARENT type_=type RIGHTPARENT));
postfixExpression:
    primaryExpression #primaryPostfixExpression
    | expr=postfixExpression LEFTSBRACKET index=expression RIGHTSBRACKET #arrayExpression
    | expr=postfixExpression LEFTPARENT (args+=assignmentExpression (COMMA args+=assignmentExpression)*)? RIGHTPARENT #funCallExpression
    | expr=postfixExpression DOT field=ID #fieldExpression
    | expr=postfixExpression TO field=ID #ptrFieldExpression
    | expr=postfixExpression DOUBLEPLUS #postincrExpression
    | expr=postfixExpression DOUBLEMINUS #postdecrExpression
;
primaryExpression:
    name=ID #varExpression
    | val=INTCST #intExpression
    | val=FLOATCST #floatExpression
    | LEFTPARENT expr=expression RIGHTPARENT #parentExpression
    | LEFTBRACKET (val+=assignmentExpression (COMMA val+=assignmentExpression)*)? RIGHTBRACKET #structureExpression
    | GENERIC LEFTPARENT expr=assignmentExpression (COMMA cases+=genericItem)+ RIGHTPARENT #genericExpression
;
genericItem:
    type_=type COLON body=assignmentExpression #typeGenericItem
    | DEFAULT COLON body=assignmentExpression #defaultGenericItem
;
