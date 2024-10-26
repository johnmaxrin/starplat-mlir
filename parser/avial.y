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

%type<astNode> type paramlist arglist arg function boolexpr declarationstmt stmt stmtlist ifstmt forstmt returnstmt forallstmt incandassignstmt assignmentstmt 


%%

prgm  : function            {root = $1;}
        | stmtlist           {root = $1;}   
        ;

function : FUNCTION IDENTIFIER LPAREN arglist RPAREN LCURLY stmtlist RCURLY     {
                                                                                        Identifier* funcName = new Identifier($2);
                                                                                        Arglist* arglist = dynamic_cast<Arglist*>($4);
                                                                                        $$ = new Function(funcName, arglist, $7);
                                                                                }
         ;

stmtlist : stmt                 {
                                  Statementlist* stmtlist = new Statementlist();
                                  stmtlist->addstmt($1);
                                  $$ = stmtlist;
                                }

         | stmtlist stmt        {
                                  Statementlist* stmtlist =dynamic_cast<Statementlist*>($1);
                                  stmtlist->addstmt($2);
                                  $$ = stmtlist;
                                }
         ;

stmt :  assignmentstmt
        |   declarationstmt             {$$ = $1;}
        |   ifstmt			{$$ = $1;}
        |   forstmt 			{$$ = $1;}
        |   returnstmt 			{$$ = $1;}
        |   forallstmt			{$$ = $1;}
        |   incandassignstmt	        {$$ = $1;}
        ;

blcstmt : LCURLY stmtlist RCURLY
        ;

declarationstmt : type IDENTIFIER SEMICLN                   {printf("Declaration statement\n");}
                | type IDENTIFIER EQUAL NUMBER SEMICLN      {$$ = new DeclarationStatement($1);}
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



arg : type IDENTIFIER           {
                                  Identifier* varname = new Identifier($2);
                                  Type* type = dynamic_cast<Type*>($1);
                                  $$ = new Arg(type, varname);
                                }
    ;


arglist : arg                   {
                                  Arglist* arglist = new Arglist();
                                  Arg* arg = dynamic_cast<Arg*>($1);
                                  arglist->addarg(arg);
                                  $$ = arglist;
                                }
        | arglist COMMA arg     {
                                  Arglist* arglist =dynamic_cast<Arglist*>($1);
                                  Arg* arg = dynamic_cast<Arg*>($3);
                                  arglist->addarg(arg);
                                  $$ = arglist;
                                }
        ;


paramlist : IDENTIFIER          {
                                  ASTNode* param = new Identifier();
                                  Paramlist* paramlist = new Paramlist();
                                  paramlist->addparam(param);
                                  $$ = paramlist;
                                  free($1);
                                  
                                } 
          | paramlist COMMA IDENTIFIER                          {
                                                                  ASTNode* param = new Identifier();
                                                                  Paramlist* paramlist = dynamic_cast<Paramlist*>($1);
                                                                  paramlist->addparam(param);
                                                                  $$ = paramlist;
                                                                  free($3);
                                                                }
          ;

type : INT              {       
                                $$ = new Type($1);
                        }

     | GRAPH            {
                                $$ = new Type($1);
                        }
     ;


%%


