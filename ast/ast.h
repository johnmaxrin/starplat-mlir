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

class ForallStatement : public ASTNode
{
    public:
        ForallStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        ForallStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitForallStmt(this);
        }
        
        ~ForallStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class IfStatement : public ASTNode
{
    public:
        IfStatement(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        IfStatement(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIfStmt(this);
        }
        
        ~IfStatement(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};


class BoolExpr : public ASTNode
{
    public:
        BoolExpr(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        BoolExpr(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitBoolExpr(this);
        }
        
        ~BoolExpr(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Incandassignstmt : public ASTNode
{
    public:
        Incandassignstmt(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        Incandassignstmt(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIncandassignstmt(this);
        }
        
        ~Incandassignstmt(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Identifier : public ASTNode
{
    public:
        Identifier(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        Identifier(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitIdentifier(this);
        }
        
        ~Identifier(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class ReturnStmt : public ASTNode
{
    public:
        ReturnStmt(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        ReturnStmt(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitReturnStmt(this);
        }
        
        ~ReturnStmt(){
            delete type;
            delete varname;
        }

        ASTNode* gettype() const{return type;}
        ASTNode* getvarname() const{return varname;}
    
    private:
        ASTNode *type;
        ASTNode *varname;
};

class Function : public ASTNode
{
    public:
        Function(ASTNode *type, ASTNode *varname)
            : type(type), varname(varname){}
        
        Function(){}

        virtual void Accept(Visitor *visitor) const override{
            visitor->visitFunction(this);
        }
        
        ~Function(){
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