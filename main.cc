#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ast/ast.h"
#include "ast/visitor.h"
#include "codegen/astdump.h"
#include "avial.tab.h"

extern int yyparse();
extern FILE *yyin;
ASTNode* root;

int main(int argc, char *argv[])
{
    root = nullptr;

    if (argc < 2)
    {
        printf("%s usage\n%s <file name>\n", argv[0], argv[0]);
        return 0;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file)
    {
        printf("Cannot open file. \n");
        return 0;
    }

    yyin = file;
    yyparse();

    CodeGen *gen = new CodeGen;

    if(root!= nullptr)
        root->Accept(gen);
    
    

    fclose(file);

    return 0;
}