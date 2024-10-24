%{
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
	#include "../ast/ast.h"
    extern int yylex();
    void yyerror(const char* s);

	extern ASTNode *root;


%}

%union {
	ASTNode* astNode;
	char* id;
}


%token<id> FUNCTION LPAREN RPAREN LCURLY RCURLY RETURN IDENTIFIER ASGN NUMBER LT GT FORALL FOR
%token<id> INT IF SEMICLN DOT IN COMMA EQUAL GRAPH PLUSEQUAL

%type<astNode> declarationstmt stmt stmtlist ifstmt forstmt returnstmt forallstmt incandassignstmt assignmentstmt 


%%

prgm  : function            {printf("FUNCTION!\n");}
        | stmtlist           {root = $1;}   
        ;

function : FUNCTION IDENTIFIER LPAREN arglist RPAREN LCURLY stmtlist RCURLY 
            ;

stmtlist : stmt         {$$ = $1;}
         | stmtlist stmt
         ;

stmt :  assignmentstmt
        |   declarationstmt     {$$ = $1;}
        |   ifstmt
        |   forstmt 
        |   returnstmt 
        |   forallstmt
        |   incandassignstmt
        ;

blcstmt : LCURLY stmtlist RCURLY
        ;

declarationstmt : type IDENTIFIER SEMICLN                   {printf("Declaration statement\n");}
                | type IDENTIFIER EQUAL NUMBER SEMICLN      {$$ = new DeclarationStatement();}
            ;

assignmentstmt : IDENTIFIER EQUAL expr SEMICLN      {printf("Assignment statement\n");}
            ;

boolexpr : IDENTIFIER LT IDENTIFIER 
         | IDENTIFIER GT IDENTIFIER
         ;

ifstmt : IF LPAREN expr RPAREN stmt         {printf("IF statement\n");}
        | IF LPAREN expr RPAREN blcstmt     {printf("IF statement\n");}
        ;


forstmt : FOR LPAREN IDENTIFIER IN expr RPAREN blcstmt      {printf("FOR statement\n");}

forallstmt : FORALL LPAREN IDENTIFIER IN expr RPAREN blcstmt    {printf("FORALL statement\n");}

expr :  IDENTIFIER 
     |  boolexpr
     |  NUMBER
     |  memberaccess
     ;

incandassignstmt : IDENTIFIER PLUSEQUAL expr SEMICLN  {printf("Increment and Assign statement\n");}
             ;

returnstmt : RETURN expr SEMICLN        {printf("Return statement\n");}
           ;

methodcall : IDENTIFIER LPAREN paramlist RPAREN 
            | IDENTIFIER LPAREN expr RPAREN 
            | IDENTIFIER LPAREN RPAREN
            ;

memberaccess : IDENTIFIER DOT methodcall
             | memberaccess DOT methodcall
             ;



arg : type IDENTIFIER
    ;


arglist : arg 
        | arglist COMMA arg
        ;


paramlist : IDENTIFIER 
          | paramlist COMMA IDENTIFIER
          ;

type : INT      
     | GRAPH 
     ;


%%


