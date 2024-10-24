#include "../ast/visitor.h"
#include <map>
#include <string>
#include <iostream>


using namespace std;

class CodeGen : public Visitor {

    public:
        CodeGen() : result(0){}
        virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt) override {
                std::cout<<"Visit Decl Statement\n";
    }


    private:
        int result;
        map<string, int> variables;
};