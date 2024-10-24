#ifndef AVIALAST
#define AVIALAST

#include "visitor.h"

class ASTNode
{
    public:
        virtual ~ASTNode() {}
        virtual void Accept(Visitor *visitor) const = 0;
};

class DeclarationStatement : public ASTNode
{
    public:
        DeclarationStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        DeclarationStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitDeclarationStmt(this);
        }
        
        ~DeclarationStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};


#endif