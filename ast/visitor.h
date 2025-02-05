#ifndef AVIALVISITOR
#define AVIALVISITOR 


class DeclarationStatement;
class TemplateDeclarationStatement;
class MemberAccessAssignment;
class ParameterAssignment;
class ForallStatement;
class IfStatement;
class BoolExpr;
class Incandassignstmt;
class Identifier;
class ReturnStmt;
class Function;
class Param;
class Paramlist;
class Arglist;
class Arg;
class Statement;
class FixedpointUntil;
class InitialiseAssignmentStmt;
class Statementlist;
class TypeExpr;
class TupleAssignment;
class TemplateType;
class Keyword;
class GraphProperties;
class Number;
class Expression;
class Methodcall;
class Memberaccess;


class Visitor
{
    public:
        virtual ~Visitor() = default;

        virtual void visitDeclarationStmt(const DeclarationStatement *declStmt) = 0;
        virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templatedeclStmt) = 0;
        virtual void visitForallStmt(const ForallStatement *forAllStmt) = 0;
        virtual void visitIfStmt(const IfStatement *ifStmt) = 0;
        virtual void visitBoolExpr(const BoolExpr *boolExpr) = 0;
        virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt) = 0;
        virtual void visitIdentifier(const Identifier *identifier) = 0;
        virtual void visitReturnStmt(const ReturnStmt *returnStmt) = 0;
        virtual void visitFunction(const Function *function) = 0;
        virtual void visitParamlist(const Paramlist *paramlist) = 0;
        virtual void visitArglist(const Arglist *arglist) = 0;
        virtual void visitTupleAssignment(const TupleAssignment *tupleAssignment) = 0;
        virtual void visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt *initialiseAssignmentStmt) = 0;
        virtual void visitArg(const Arg *arg) = 0;
        virtual void visitMemberAccessAssignment(const MemberAccessAssignment *memberAccessAssignment) = 0;
        virtual void visitStatement(const Statement *stmt) = 0;
        virtual void visitStatementlist(const Statementlist *stmt) = 0;
        virtual void visitFixedpointUntil(const FixedpointUntil *fixedpointuntil) = 0;
        virtual void visitTemplateType(const TemplateType *templatetype) = 0;
        virtual void visitParameterAssignment(const ParameterAssignment *parameterAssignment) = 0;
        virtual void visitParam(const Param *param) = 0;
        virtual void visitType(const TypeExpr *type) = 0;
        virtual void visitKeyword(const Keyword *keyword) = 0;
        virtual void visitGraphProperties(const GraphProperties *graphproperties) = 0;
        virtual void visitNumber(const Number *number) = 0;
        virtual void visitExpression(const Expression *expr) = 0;
        virtual void visitMethodcall(const Methodcall *methodcall) = 0;
        virtual void visitMemberaccess(const Memberaccess *methodcall) = 0;
    
};

#endif