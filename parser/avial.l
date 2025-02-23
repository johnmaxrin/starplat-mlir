%{
    #include "../../ast/ast.h"
    #include "avial.tab.h"

    extern YYSTYPE yylval;
%}

%%

"//".*                  { /* Ignore single-line comments */ }

"function"              {return FUNCTION;}
"return"                {return RETURN;}        
"if"                    {return IF;}
"forall"                {return FORALL;}
"for"                   {return FOR;}
"Graph"                 {yylval.id = strdup(yytext); return GRAPH;}
"int"                   {yylval.id = strdup(yytext); return INT;}
"propNode"              {yylval.id = strdup(yytext); return PROPNODE; }
"propEdge"              {yylval.id = strdup(yytext); return PROPEDGE; }
"False"                 {yylval.id = strdup(yytext); return FALSE; }
"True"                 { yylval.id = strdup(yytext); return TRUE; }
"INF"                   {yylval.id = strdup(yytext); return INF; }
"fixedPoint"            {yylval.id = strdup(yytext); return FIXEDPOINT; }
"until"                 {yylval.id = strdup(yytext); return UNTIL; }
"edge"                  {yylval.id = strdup(yytext); return EDGE; }



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
":"                     {return COLON;}
"=="                    {return EQUALS;}
"+"                     {return PLUS;}

[ \t\n]+                ;
#.*                    { /* Ignore single-line comments */ }

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