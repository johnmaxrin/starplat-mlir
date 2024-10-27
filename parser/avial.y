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

%type<astNode>  methodcall blcstmt memberaccess expr type paramlist arglist arg function boolexpr declarationstmt stmt stmtlist ifstmt forstmt returnstmt forallstmt incandassignstmt assignmentstmt 


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

blcstmt : LCURLY stmtlist RCURLY        {$$ = $2;}
        ;

declarationstmt : type IDENTIFIER SEMICLN                   {printf("Declaration statement\n");}

                | type IDENTIFIER EQUAL NUMBER SEMICLN  {
                                                                Identifier* identifier = new Identifier($2);
                                                                Number* number =   new Number($4);
                                                                $$ = new DeclarationStatement($1, identifier, number);
                                                        
                                                        }
            ;

assignmentstmt : IDENTIFIER EQUAL expr SEMICLN      {$$ = new Incandassignstmt();}
            ;

boolexpr : IDENTIFIER LT IDENTIFIER 		{
                                                  Identifier *id1 = new Identifier($1);
                                                  Identifier *id2 = new Identifier($3);
                                                  $$ = new BoolExpr(id1, $2, id2);
                                                }

         | IDENTIFIER GT IDENTIFIER             {
                                                  Identifier *id1 = new Identifier($1);
                                                  Identifier *id2 = new Identifier($3);
                                                  $$ = new BoolExpr(id1, $2, id2);
                                                }
         ;

ifstmt : IF LPAREN expr RPAREN stmt         {$$ = new IfStatement($3, $5);}
        | IF LPAREN expr RPAREN blcstmt     {$$ = new IfStatement($3, $5);}
        ;


forstmt : FOR LPAREN IDENTIFIER IN expr RPAREN blcstmt      { $$ = new ForallStatement();}

forallstmt : FORALL LPAREN IDENTIFIER IN expr RPAREN blcstmt    { 
                                                                  Identifier* identifier = new Identifier($3);
                                                                  $$ = new ForallStatement(identifier, $5, $7);
                                                                }

expr :  IDENTIFIER              {$$ = new Identifier($1);} 
     |  boolexpr                {$$ = $1;}
     |  NUMBER                  {$$ = new Number($1);}
     |  memberaccess            {$$ = $1;}
     ;

incandassignstmt : IDENTIFIER PLUSEQUAL expr SEMICLN  {
                                                        Identifier* identifier = new Identifier($1);
                                                        $$ = new Incandassignstmt(identifier, $2, $3);
                                                      }
             ;

returnstmt : RETURN expr SEMICLN        {$$ = new ReturnStmt($2);}
           ;

methodcall : IDENTIFIER LPAREN paramlist RPAREN {
                                                  Identifier* identifier = new Identifier($1);
                                                  $$ = new Methodcall(identifier, $3);
                                                }

            | IDENTIFIER LPAREN expr RPAREN     {
                                                  Identifier* identifier = new Identifier($1);
                                                  $$ = new Methodcall(identifier, $3);
                                                }

            | IDENTIFIER LPAREN RPAREN          {
                                                  Identifier* identifier = new Identifier($1);
                                                  $$ = new Methodcall(identifier);
                                                }
            ;

memberaccess : IDENTIFIER DOT methodcall        {
                                                  Identifier* identifier = new Identifier($1);
                                                  $$ = new Memberaccess(identifier, $3);
                                                }
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


