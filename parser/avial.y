%{
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
	  #include "../../ast/ast.h"
    extern int yylex();
    void yyerror(const char* s);

	extern ASTNode *root;


%}

%union {
	ASTNode* astNode;
	char* id;
}


%token<id> FUNCTION LPAREN RPAREN LCURLY RCURLY RETURN IDENTIFIER ASGN NUMBER LT GT FORALL FOR EQUALS EDGE
%token<id> INT IF SEMICLN DOT IN COMMA EQUAL GRAPH PLUSEQUAL PROPNODE PROPEDGE FALSE INF FIXEDPOINT UNTIL COLON PLUS

%type<astNode>  methodcall blcstmt memberaccess expr type paramlist arglist arg function boolexpr declarationstmt stmt 
stmtlist ifstmt forstmt returnstmt forallstmt incandassignstmt assignment initializestmt fixedPointStmt tuppleAssignmentstmt
addExpr KEYWORDS


%%

prgm  : function            {root = $1;}
        | stmtlist           {root = $1;}   
        ;

function : FUNCTION IDENTIFIER LPAREN arglist RPAREN LCURLY stmtlist RCURLY     {
                                                                                        Identifier* funcName = new Identifier($2);
                                                                                        Arglist* arglist = static_cast<Arglist*>($4);
                                                                                        $$ = new Function(funcName, arglist, $7);
                                                                                }
         ;

stmtlist : stmt                 {
                                  Statementlist* stmtlist = new Statementlist();
                                  stmtlist->addstmt($1);
                                  $$ = stmtlist;
                                }

         | stmtlist stmt        {
                                  Statementlist* stmtlist =static_cast<Statementlist*>($1);
                                  stmtlist->addstmt($2);
                                  $$ = stmtlist;
                                }
         ;

stmt :  assignment SEMICLN
        |   declarationstmt             {$$ = $1;}
        |   ifstmt			                {$$ = $1;}
        |   forstmt 			              {$$ = $1;}
        |   returnstmt 			            {$$ = $1;}
        |   forallstmt			            {$$ = $1;}
        |   incandassignstmt	          {$$ = $1;}
        |   templateDecl                {}
        |   /*epsilon*/                 {$$ = nullptr;}
        |   memberaccessstmt            {}
        |   initializestmt              {}
        |   memberaccess EQUAL expr SEMICLN     {}
        |   fixedPointStmt              {}
        |   tuppleAssignmentstmt        {}
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

templateDecl : templateType IDENTIFIER SEMICLN { }
              ;

assignment : IDENTIFIER EQUAL expr      {$$ = new Incandassignstmt();}
            ;

initializestmt : type IDENTIFIER EQUAL expr SEMICLN {}

paramAssignment : IDENTIFIER EQUAL KEYWORDS {}

fixedPointStmt : FIXEDPOINT UNTIL LPAREN IDENTIFIER COLON expr RPAREN LCURLY stmtlist RCURLY         {}

tuppleAssignmentstmt : LT expr COMMA expr GT EQUAL LT expr COMMA expr GT SEMICLN               {}

boolexpr : expr LT expr 		{}

         | expr GT expr             {}

         | expr EQUALS expr           {} 
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
     |  KEYWORDS                {}
     |  methodcall              {}
     |  addExpr                 {}
     ;

addExpr : expr PLUS expr      {}

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
             | memberaccess DOT methodcall      {}
             | IDENTIFIER DOT IDENTIFIER        {}
             ;

memberaccessstmt : memberaccess SEMICLN {}

arg : type IDENTIFIER           {
                                  Identifier* varname = new Identifier($2);
                                  TypeExpr* type = static_cast<TypeExpr*>($1);
                                  $$ = new Arg(type, varname);
                                }
      | templateType IDENTIFIER {}
      
      | /*epsilon*/             {
                                    $$ = nullptr;
                                }
      | IDENTIFIER EQUAL IDENTIFIER   {}
    
      ;


arglist : arg                   {
                                  Arglist* arglist = new Arglist();
                                  Arg* arg = static_cast<Arg*>($1);
                                  arglist->addarg(arg);
                                  $$ = arglist;
                                }
        
        | arglist COMMA arg     
                                {
                                  Arglist* arglist =static_cast<Arglist*>($1);
                                  Arg* arg = static_cast<Arg*>($3);
                                  arglist->addarg(arg);
                                  $$ = arglist;
                                }
        ;


param : expr 
         | paramAssignment 
         ;


paramlist : param                                          {}
          | paramlist COMMA param                          {}
          ;

templateType : properties LT type GT ;

properties : PROPEDGE 
             | PROPNODE
             ;


type : INT              {       
                                $$ = new TypeExpr($1);
                        }

     | GRAPH            {
                                $$ = new TypeExpr($1);
                        }

     |  EDGE            {
                                $$ = new TypeExpr($1);
                        }
     ;

KEYWORDS : FALSE         {
                                $$ = new Keyword($1);
                         }

          | INF          {
                                $$ = new Keyword($1);
                         }
          ;


%%


