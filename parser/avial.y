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

%type<astNode> function boolexpr declarationstmt stmt stmtlist ifstmt forstmt returnstmt forallstmt incandassignstmt assignmentstmt 


%%

prgm  : function            {root = $1;}
        | stmtlist           {root = $1;}   
        ;

function : FUNCTION IDENTIFIER LPAREN arglist RPAREN LCURLY stmtlist RCURLY     {$$ = new Function();}
            ;

stmtlist : stmt         {$$ = $1;}
         | stmtlist stmt
         ;

stmt :  assignmentstmt
        |   declarationstmt     {$$ = $1;}
        |   ifstmt				{$$ = $1;}
        |   forstmt 			{$$ = $1;}
        |   returnstmt 			{$$ = $1;}
        |   forallstmt			{$$ = $1;}
        |   incandassignstmt	{$$ = $1;}
        ;

blcstmt : LCURLY stmtlist RCURLY
        ;

declarationstmt : type IDENTIFIER SEMICLN                   {printf("Declaration statement\n");}
                | type IDENTIFIER EQUAL NUMBER SEMICLN      {$$ = new DeclarationStatement();}
            ;

assignmentstmt : IDENTIFIER EQUAL expr SEMICLN      {$$ = new Incandassignstmt();}
            ;

boolexpr : IDENTIFIER LT IDENTIFIER 		{$$ = new BoolExpr();}
         | IDENTIFIER GT IDENTIFIER             {$$ = new BoolExpr();}
         ;

ifstmt : IF LPAREN expr RPAREN stmt         {$$ = new IfStatement();}
        | IF LPAREN expr RPAREN blcstmt     {$$ = new IfStatement();}
        ;


forstmt : FOR LPAREN IDENTIFIER IN expr RPAREN blcstmt      {$$ = new ForallStatement();}

forallstmt : FORALL LPAREN IDENTIFIER IN expr RPAREN blcstmt    {$$ = new ForallStatement();}

expr :  IDENTIFIER 
     |  boolexpr
     |  NUMBER
     |  memberaccess
     ;

incandassignstmt : IDENTIFIER PLUSEQUAL expr SEMICLN  {$$ = new Incandassignstmt();}
             ;

returnstmt : RETURN expr SEMICLN        {$$ = new ReturnStmt();}
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


