#ifndef AVIALAST
#define AVIALAST

#include "visitor.h"
#include <vector>
#include <set>
#include <iostream>
#include <mlir/IR/SymbolTable.h>

using namespace std;
using namespace mlir;

class ASTNode
{
public:
    virtual ~ASTNode() {}
    virtual void Accept(Visitor *visitor) const = 0;
    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const = 0;
};


enum ExpressionKind{
    KIND_NUMBER,
    KIND_IDENTIFIER,
    KIND_KEYWORD,
    KIND_BOOLEXPR,
    KIND_MEMBERACCESS,
    KIND_METHODCALL,
    KIND_ADDOP
};



class Identifier : public ASTNode
{
public:
    Identifier(char *name)
        : name_(name) {}

    Identifier() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitIdentifier(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitIdentifier(this, symbolTable);
    }

    ~Identifier()
    {
        delete name_;
    }

    char *getname() const { return name_; }

private:
    char *name_;
};

class GraphProperties : public ASTNode
{
public:
    GraphProperties(char *properties)
        : properties_(properties) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitGraphProperties(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitGraphProperties(this, symbolTable);
    }

    ~GraphProperties()
    {
        delete properties_;
    }

    char *getPropertyType() const { return properties_; }

private:
    char *properties_;
};

class Statement : public ASTNode
{
public:
    Statement() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitStatement(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitStatement(this, symbolTable);
    }

    ~Statement()
    {
        delete statement;
    }

    ASTNode *getstatement() const { return statement; }

private:
    ASTNode *statement;
};

class Methodcall : public ASTNode
{
public:
    Methodcall(Identifier *identifier, ASTNode *paramlist)
        : identifier_(identifier), paramlist_(paramlist), _isBuiltin(checkIfBuiltin(identifier)) {}

    Methodcall(Identifier *identifier)
        : identifier_(identifier), paramlist_(nullptr), _isBuiltin(checkIfBuiltin(identifier)) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitMethodcall(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitMethodcall(this, symbolTable);
    }

    ~Methodcall()
    {
        delete identifier_;
        delete paramlist_;
    }

    const Identifier *getIdentifier() const { return identifier_; }
    const ASTNode *getParamLists() const { return paramlist_; }
    bool getIsBuiltin() const { return _isBuiltin; }

private:
    const Identifier * identifier_;
    const ASTNode * paramlist_;
    bool  _isBuiltin;

    static bool checkIfBuiltin(const Identifier *id)
    {
        static const std::set<std::string> builtins = {
            "print", "attachNodeProperty", "filter", "get_edge", "neighbors", "nodes", "Min" // Add more built-in methods
        };
        return builtins.find(id->getname()) != builtins.end();
    }
};

class TupleAssignment : public ASTNode
{
public:
    TupleAssignment(ASTNode *lhsexpr1, ASTNode *lhsexpr2, ASTNode *rhsexpr1, ASTNode *rhsexpr2)
        : lhsexpr1_(lhsexpr1), lhsexpr2_(lhsexpr2), rhsexpr1_(rhsexpr1), rhsexpr2_(rhsexpr2) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitTupleAssignment(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitTupleAssignment(this, symbolTable);
    }

    ~TupleAssignment()
    {
        delete lhsexpr1_;
        delete lhsexpr2_;
        delete rhsexpr1_;
        delete rhsexpr2_;
    }

    const ASTNode *getlhsexpr1() const { return lhsexpr1_; }
    const ASTNode *getlhsexpr2() const { return lhsexpr2_; }
    const ASTNode *getrhsexpr1() const { return rhsexpr1_; }
    const ASTNode *getrhsexpr2() const { return rhsexpr2_; }

private:
    const ASTNode *lhsexpr1_;
    const ASTNode *lhsexpr2_;
    const ASTNode *rhsexpr1_;
    const ASTNode *rhsexpr2_;
};


class MemberacceessStmt : public ASTNode
{
    public:
        MemberacceessStmt(ASTNode *memberAccess) : memberAccess_(memberAccess) {}

        virtual void Accept(Visitor *visitor) const override
        {
            visitor->visitMemberaccessStmt(this);
        }

        virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
        {
            visitor->visitMemberaccessStmt(this, symbolTable);
        }

        const ASTNode *getMemberAccess() const { return memberAccess_; }

        ~MemberacceessStmt()
        {
            delete memberAccess_;
        }

    private:
        ASTNode *memberAccess_;
};


class Memberaccess : public ASTNode
{
public:
    Memberaccess(Identifier *identifier1, ASTNode *methodcall)
        : identifier1_(identifier1), methodcall_(methodcall), identifier2_(nullptr), memberaccessNode_(nullptr) {}

    Memberaccess(Identifier *identifier1, Identifier *identifier2)
        : identifier1_(identifier1), methodcall_(nullptr), identifier2_(identifier2), memberaccessNode_(nullptr) {}

    Memberaccess(ASTNode *memberaccessNode, ASTNode *methodcall)
        : identifier1_(nullptr), methodcall_(methodcall), identifier2_(nullptr), memberaccessNode_(memberaccessNode) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitMemberaccess(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitMemberaccess(this, symbolTable);
    }

    ~Memberaccess()
    {
        delete identifier1_;
        delete identifier2_;
        delete memberaccessNode_;
        delete methodcall_;
    }

    const Identifier *getIdentifier() const { return identifier1_; }
    const Identifier *getIdentifier2() const { return identifier2_; }
    const ASTNode *getMethodCall() const { return methodcall_; }
    const ASTNode *getMemberAccessNode() const { return memberaccessNode_; }

private:
    const Identifier *identifier1_;
    const Identifier *identifier2_;
    const ASTNode *methodcall_;
    const ASTNode *memberaccessNode_;
};

class Statementlist : public ASTNode
{
public:
    Statementlist() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitStatementlist(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        for (ASTNode *stmt : statementlist)
        {
            stmt->Accept(visitor, symbolTable);
        }
    }



    ~Statementlist()
    {
        for (ASTNode *param : statementlist)
            delete param;
    }

    const vector<ASTNode *> getStatementList() const { return statementlist; }
    void addstmt(ASTNode *param) { statementlist.push_back(param); }

private:
    vector<ASTNode *> statementlist;
};

class DeclarationStatement : public ASTNode
{
public:
    DeclarationStatement(ASTNode *type, ASTNode *identifier, ASTNode *number)
        : type(type), varname(identifier), number(number) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitDeclarationStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitDeclarationStmt(this, symbolTable);
    }

    ~DeclarationStatement()
    {
        delete type;
        delete varname;
        delete number;
    }

    ASTNode *gettype() const { return type; }
    ASTNode *getvarname() const { return varname; }
    ASTNode *getnumber() const { return number; }

private:
    ASTNode *type;
    ASTNode *varname;
    ASTNode *number;
};


class InitialiseAssignmentStmt : public ASTNode
{
public:
    InitialiseAssignmentStmt(ASTNode *type,  ASTNode *identifier, ASTNode *expr)
        : type_(type), identifier_(identifier), expr_(expr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitInitialiseAssignmentStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitInitialiseAssignmentStmt(this, symbolTable);
    }

    ~InitialiseAssignmentStmt()
    {
        delete type_;
        delete identifier_;
        delete expr_;
    }

    ASTNode *gettype() const { return type_; }
    ASTNode *getidentifier() const { return identifier_; }
    ASTNode *getexpr() const { return expr_; }

private:
    ASTNode *type_;
    ASTNode *identifier_;
    ASTNode *expr_;
};

class FixedpointUntil : public ASTNode
{
public:
    FixedpointUntil(ASTNode *identifier , ASTNode *expr, ASTNode *stmtlist)
        : expr_(expr), identifier_(identifier),  stmtlist_(stmtlist)  {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitFixedpointUntil(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
       visitor->visitFixedpointUntil(this, symbolTable); 
    }

    ASTNode *getstmtlist() const{ return stmtlist_; }
    ASTNode *getidentifier() const{ return identifier_; }
    ASTNode *getexpr() const{ return expr_; }

    ~FixedpointUntil()
    {
        delete stmtlist_;
        delete identifier_;
        delete expr_;
    }

private:
    ASTNode *expr_;
    ASTNode *identifier_;
    ASTNode *stmtlist_;
};

class ParameterAssignment : public ASTNode
{
public:
    ParameterAssignment(ASTNode *identifier, ASTNode *keyword)
        : identifier_(identifier), keyword_(keyword) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitParameterAssignment(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitParameterAssignment(this, symbolTable);
    }

    ~ParameterAssignment()
    {
        delete identifier_;
        delete keyword_;
    }

    ASTNode *getidentifier() const { return identifier_; }
    ASTNode *getkeyword() const { return keyword_; }

private:
    ASTNode *identifier_;
    ASTNode *keyword_;
};

class TemplateDeclarationStatement : public ASTNode
{
public:
    TemplateDeclarationStatement(ASTNode *type, ASTNode *identifier)
        : type(type), varname(identifier) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitTemplateDeclarationStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitTemplateDeclarationStmt(this, symbolTable);
    }

    ~TemplateDeclarationStatement()
    {
        delete type;
        delete varname;
    }

    ASTNode *gettype() const { return type; }
    ASTNode *getvarname() const { return varname; }

private:
    ASTNode *type;
    ASTNode *varname;
};

class ForallStatement : public ASTNode
{
public:
    ForallStatement(Identifier *loopVar, ASTNode *expr, ASTNode *stmtlist)
        : loopVar_(loopVar), expr_(expr), stmtlist_(stmtlist) {}

    ForallStatement() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitForallStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitForallStmt(this, symbolTable);
    }

    ~ForallStatement()
    {
        delete loopVar_;
        delete expr_;
        delete stmtlist_;
    }

    Identifier *getLoopVar() const { return loopVar_; }
    ASTNode *getexpr() const { return expr_; }
    ASTNode *getstmtlist() const { return stmtlist_; }

private:
    Identifier *loopVar_;
    ASTNode *expr_;
    ASTNode *stmtlist_;
};

class IfStatement : public ASTNode
{
public:
    IfStatement(ASTNode *expr, ASTNode *stmt)
        : expr(expr), stmt(stmt) {}

    IfStatement() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitIfStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitIfStmt(this, symbolTable);
    }

    ~IfStatement()
    {
        delete expr;
        delete stmt;
    }

    ASTNode *getexpr() const { return expr; }
    ASTNode *getstmt() const { return stmt; }

private:
    ASTNode *expr;
    ASTNode *stmt;
};

class BoolExpr : public ASTNode
{
public:
    BoolExpr(ASTNode *expr1, char *op, ASTNode *expr2)
        : expr1_(expr1), op_(op), expr2_(expr2) {}

    BoolExpr(ASTNode *expr1, char *op) 
        : expr1_(expr1), op_(op), expr2_(nullptr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitBoolExpr(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitBoolExpr(this, symbolTable);
    }

    ~BoolExpr()
    {
        delete expr1_;
        delete expr2_;
        delete op_;
    }

    ASTNode*getExpr1() const { return expr1_; }
    ASTNode *getExpr2() const { return expr2_; }
    char *getop() const { return op_; }

private:
    ASTNode *expr1_;
    ASTNode *expr2_;
    char *op_;
};


class Add : public ASTNode
{
public:

    Add(ASTNode *operand1, char *op, ASTNode *operand2) 
        : operand1_(operand1), op_(op), operand2_(operand2_) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitAdd(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitAdd(this, symbolTable);
    }

    ~Add()
    {
        delete operand1_;
        delete operand2_;
        delete op_;
    }

    ASTNode *getOperand1() const { return operand1_; }
    ASTNode *getOperand2() const { return operand2_; }
    char *getop() const { return op_; }

private:
    ASTNode *operand1_;
    ASTNode *operand2_;
    char *op_;
};


class Incandassignstmt : public ASTNode
{
public:
    Incandassignstmt(Identifier *identifier, char *op, ASTNode *expr)
        : identifier(identifier), op(op), expr(expr) {}

    Incandassignstmt() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitIncandassignstmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitIncandassignstmt(this, symbolTable);
    }

    ~Incandassignstmt()
    {
        delete identifier;
        delete op;
        delete expr;
    }

    Identifier *getIdentifier() const { return identifier; }
    ASTNode *getexpr() const { return expr; }

private:
    Identifier *identifier;
    char *op;
    ASTNode *expr;
};

class Number : public ASTNode
{
public:
    Number(char *number)
        : number_(atoi(number)) {}

    Number() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitNumber(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitNumber(this, symbolTable);
    }

    int getnumber() const { return number_; }

private:
    int number_;
};

class ReturnStmt : public ASTNode
{
public:
    ReturnStmt(ASTNode *expr)
        : expr(expr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitReturnStmt(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitReturnStmt(this, symbolTable);
    }

    ~ReturnStmt()
    {
        delete expr;
    }

    ASTNode *getexpr() const { return expr; }

private:
    ASTNode *expr;
};

class Function : public ASTNode
{
public:
    Function(Identifier *functionname, Arglist *arglist, ASTNode *stmtlist)
        : functionname(functionname), arglist(arglist), stmtlist(stmtlist)
    {

        funcName = functionname->getname();
    }

    // Function(){}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitFunction(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitFunction(this, symbolTable);
    }

    ~Function()
    {
        delete functionname;
        delete stmtlist;
    }

    ASTNode *getfuncname() const { return functionname; }
    Arglist *getparams() const { return arglist; }
    ASTNode *getstmtlist() const { return stmtlist; }
    string getfuncNameIdentifier() const { return funcName; } // Returns the function name as a string.

private:
    ASTNode *functionname;
    Arglist *arglist;
    ASTNode *stmtlist;
    string funcName;
};

class Paramlist : public ASTNode
{
public:
    Paramlist() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitParamlist(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitParamlist(this, symbolTable);
    }

    ~Paramlist()
    {
        for (Param *param : paramlist)
            delete param;
    }

    const vector<Param *> getParamList() const { return paramlist; }
    void addparam(Param *param) { paramlist.push_back(param); }

private:
    vector<Param *> paramlist;
};

class TemplateType : public ASTNode
{
public:
    TemplateType(GraphProperties *graphprop, TypeExpr *type) : graphproperties_(graphprop), type_(type) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitTemplateType(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitTemplateType(this, symbolTable);
    }

    const GraphProperties *getGraphPropNode() const { return graphproperties_; }
    const TypeExpr *getType() const { return type_; }

private:
    GraphProperties *graphproperties_;
    TypeExpr *type_;
};

class TypeExpr : public ASTNode
{
public:
    TypeExpr(char *type) : type_(type) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitType(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitType(this, symbolTable);
    }

    ~TypeExpr()
    {
        delete type_;
    }

    const char *getType() const { return type_; }

private:
    char *type_;
};

class MemberAccessAssignment : public ASTNode
{
public:
    MemberAccessAssignment(ASTNode *memberAccess, ASTNode *expr) : memberAccess_(memberAccess), expr_(expr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitMemberAccessAssignment(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitMemberAccessAssignment(this, symbolTable);
    }

    ~MemberAccessAssignment()
    {
        delete memberAccess_;
        delete expr_;
    }

    const ASTNode *getMemberAccess() const { return memberAccess_; }
    const ASTNode *getExpr() const {return expr_;}

private:
    ASTNode *memberAccess_;
    ASTNode *expr_;
};

class Keyword : public ASTNode
{
public:
    Keyword(char *keyword) : keyword_(keyword) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitKeyword(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitKeyword(this, symbolTable);
    }

    ~Keyword()
    {
        delete keyword_;
    }

    const char *getKeyword() const { return keyword_; }

private:
    char *keyword_;
};

class Arg : public ASTNode
{
public:
    Arg(TypeExpr *type, Identifier *identifier)
        : type(type), varname(identifier), templatetype(nullptr) {}
    
    Arg(TemplateType *type, Identifier *identifier)
        : templatetype(type), varname(identifier), type(nullptr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitArg(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitArg(this, symbolTable);
    }

    ~Arg()
    {
        delete type;
        delete varname;
    }

    const TypeExpr *getType() const { return type; }
    const TemplateType *getTemplateType() const { return templatetype; }
    const Identifier *getVarName() const { return varname; }

private:
    TypeExpr *type;
    TemplateType *templatetype;
    Identifier *varname;
};

class Arglist : public ASTNode
{
public:
    Arglist() {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitArglist(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitArglist(this, symbolTable);
    }

    ~Arglist()
    {
        for (Arg *arg : arglist)
            delete arg;
    }

    const vector<Arg *> getArgList() const { return arglist; }
    void addarg(Arg *arg) { arglist.push_back(arg); }

private:
    vector<Arg *> arglist;
};




class Param : public ASTNode
{
public:
    Param(Expression *expr)
        : expr_(expr), paramAssignment_(nullptr) {}

    Param(ParameterAssignment *paramAssignment)
        : paramAssignment_(paramAssignment), expr_(nullptr) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitParam(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitParam(this, symbolTable);
    }


    ~Param()
    {
        delete expr_;
        delete paramAssignment_;
    }

    const Expression *getExpr() const { return expr_; }
    const ParameterAssignment *getParamAssignment() const { return paramAssignment_; }

private:
    Expression *expr_;
    ParameterAssignment *paramAssignment_;
};



class Expression : public ASTNode
{
public:
    Expression(ASTNode *node, ExpressionKind kind) : node_(node), kind_(kind) {}

    virtual void Accept(Visitor *visitor) const override
    {
        visitor->visitExpression(this);
    }

    virtual void Accept(MLIRVisitor *visitor, mlir::SymbolTable *symbolTable) const override
    {
        visitor->visitExpression(this, symbolTable);
    }

    const ASTNode *getExpression() const { return node_; }
    const ExpressionKind getKind() const { return kind_; }

private:
    ASTNode *node_;
    ExpressionKind kind_;

};

#endif