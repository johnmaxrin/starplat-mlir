#ifndef AVIALVISITOR
#define AVIALVISITOR 


class DeclarationStatement;

class Visitor
{
    public:
        virtual ~Visitor() = default;

        virtual void visitDeclarationStmt(const DeclarationStatement *declstmt) = 0;
};

#endif