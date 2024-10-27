%{
    #include "../ast/ast.h"
    #include "avial.tab.h"

    extern YYSTYPE yylval;
%}

%%

"function"              {return FUNCTION;}
"return"                {return RETURN;}        
"if"                    {return IF;}
"forall"                {return FORALL;}
"for"                   {return FOR;}
"Graph"                 {yylval.id = strdup(yytext); return GRAPH;}
"int"                   {yylval.id = strdup(yytext); return INT;}


"{"                     {return LCURLY;}
"}"                     {return RCURLY;}
"("                     {return LPAREN;}
")"                     {return RPAREN;} 
"<"                     {return LT;}
">"                     {return GT;}
"="                     {return EQUAL;}
";"                     {return SEMICLN;}
"."                     {return DOT;}
"in"                    {return IN;}
","                     {return COMMA;}
"+="                    {return PLUSEQUAL;}

[ \t\n]+                ;

[a-zA-Z_][a-zA-Z0-9_]*  {yylval.id = strdup(yytext); return IDENTIFIER;}
[0-9]*                  {yylval.id = strdup(yytext); return NUMBER;}

%%

int yywrap(void)
{
    return 1;
}


void yyerror(const char* s) {
    fprintf(stderr, "Error: %s\n", s);
}