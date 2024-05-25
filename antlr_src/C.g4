//===------------- Copyright 2024 Dylan Leothaud --------------------------===//
//
// Under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

grammar C;

WHITESPACE: [ \t\r\n] -> channel(HIDDEN);
LINECOMMENT: '//' ~[\n]* -> channel(HIDDEN);
BLOCKCOMMENT: '/*' .*? '*/' -> channel(HIDDEN);

LEFTPARENT: '(';
RIGHTPAREN: ')';
LEFTBRACKET: '{';
RIGHTBRACKET: '}';
LEFTSBRACKET: '[';
RIGHTSBRACKET: ']';

COMMA: ',';
SEMI: ';';
COLON: ':';

DOUBLEPLUS: '++';
DOUBLEMINUX: '--';
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

ID: [a-zA-Z] [a-zA-Z0-9_]*;
FLOATCST: [0-9]+ '.' [0-9]+;
INTCST: [0-9]+;
STRINGCST: '"' ~["] '"';