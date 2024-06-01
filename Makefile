antlr:
	mkdir -p antlr_gen
	antlr4 antlr_src/MyC.g4 -o antlr_gen -Xexact-output-dir -no-listener -visitor -Dlanguage=Cpp
