#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"
#include "includes/StarPlatTypes.h"

#include "mlir/IR/Builders.h"  // For mlir::OpBuilder
#include "mlir/IR/Operation.h" // For mlir::Operation
#include "mlir/IR/Dialect.h"   // For mlir::Dialect

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

class StarPlatCodeGen : public Visitor
{

public:
    StarPlatCodeGen() : result(0),
                        context(),
                        builder(&context),
                        module(mlir::ModuleOp::create(builder.getUnknownLoc()))
    {
        // Load Dialects here.
        context.getOrLoadDialect<mlir::starplat::StarPlatDialect>();
    }

    virtual void visitDeclarationStmt(const DeclarationStatement *dclstmt) override
    {
    }

    virtual void visitTemplateDeclarationStmt(const TemplateDeclarationStatement *templateDeclStmt)
    {
    }

    virtual void visitTemplateType(const TemplateType *templateType)
    {
    }

    virtual void visitForallStmt(const ForallStatement *forAllStmt) override
    {
    }

    virtual void visitIfStmt(const IfStatement *ifStmt) override
    {
    }

    virtual void visitBoolExpr(const BoolExpr *boolExpr) override
    {
    }

    virtual void visitIncandassignstmt(const Incandassignstmt *incandassignstmt) override
    {
    }

    virtual void visitIdentifier(const Identifier *identifier) override
    {
    }

    virtual void visitReturnStmt(const ReturnStmt *returnStmt) override
    {
    }

    virtual void visitParameterAssignment(const ParameterAssignment *paramAssignment)
    {
    }

    virtual void visitParam(const Param *param)
    {
    }

    virtual void visitTupleAssignment(const TupleAssignment *tupleAssignment)
    {
    }

    virtual void visitFunction(const Function *function) override
    {
        llvm::StringRef value = function->getfuncNameIdentifier();

        auto grty = mlir::starplat::GraphType::get(builder.getContext());
        auto ndty = mlir::starplat::NodeType::get(builder.getContext());
        auto propndty = mlir::starplat::PropNodeType::get(builder.getContext(), builder.getI32Type());
        auto propedty = mlir::starplat::PropEdgeType::get(builder.getContext(), builder.getI32Type());

        auto funcType = builder.getFunctionType({grty, propndty, propedty, ndty}, {});
        llvm::ArrayRef<mlir::NamedAttribute> attrs;
        llvm::ArrayRef<mlir::DictionaryAttr> args;

        // auto funcbl = builder.create<mlir::starplat::FuncOp>(builder.getUnknownLoc(),value);
        mlir::OperationState state(builder.getUnknownLoc(), "starplat.func");

        auto arg1 = builder.getStringAttr("g");
        auto arg2 = builder.getStringAttr("dist");
        auto arg3 = builder.getStringAttr("weight");
        auto arg4 = builder.getStringAttr("src");

        auto argNames = builder.getArrayAttr({arg1, arg2, arg3, arg4});

        auto funcbl = builder.create<mlir::starplat::FuncOp>(builder.getUnknownLoc(), function->getfuncNameIdentifier(), funcType, argNames);

        // propNode <int> modified;
        auto type = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type());
        auto typeAttr = ::mlir::TypeAttr::get(type);
        auto resType = builder.getI32Type();
        auto declare = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), resType, typeAttr);

        // propNode <int> modifiednxt;
        auto type2 = builder.getType<mlir::starplat::PropNodeType>(builder.getI32Type());
        auto typeAttr2 = ::mlir::TypeAttr::get(type);
        auto resType2 = builder.getI32Type();
        auto declare2 = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), resType2, typeAttr2);

        auto boolType = builder.getI1Type();

        module.push_back(funcbl);

        auto &entryBlock = funcbl.getBody().emplaceBlock();

        for (auto arg : funcType.getInputs())
            entryBlock.addArgument(arg, builder.getUnknownLoc());

        builder.setInsertionPointToStart(&entryBlock);

        entryBlock.push_back(declare);
        entryBlock.push_back(declare2);

        auto infAttr = builder.getStringAttr("INF");
        auto falseAttr = builder.getStringAttr("False");
        auto trueAttr = builder.getStringAttr("True");

        auto INFSSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI32Type(), infAttr);
        auto FALSESSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI1Type(), falseAttr);
        auto TRUESSA = builder.create<mlir::starplat::ConstOp>(builder.getUnknownLoc(), builder.getI1Type(), trueAttr);

        // dist = INF
        auto lhs = entryBlock.getArgument(1);
        auto rhs = INFSSA.getResult();
        auto assign1 = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), lhs, rhs);

        // modified = False
        auto value1 = declare.getResult();
        auto assign2 = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), value1, FALSESSA.getResult());

        // modified_nxt = False
        auto value2 = declare2.getResult();
        auto assign3 = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), value2, FALSESSA.getResult());


        llvm::SmallVector<mlir::Value, 2> operands = {entryBlock.getArgument(1), declare.getResult(), declare2.getResult()};
        
        // g.attachNodeProperty(dist=INF, modified = False, modified_nxt = False );
        auto attachnodeprop = builder.create<mlir::starplat::AttachNodePropertyOp>(builder.getUnknownLoc(), operands);

        // src.modified = True; 
        auto setNode1 = builder.create<mlir::starplat::SetNodePropertyOp>(builder.getUnknownLoc(),entryBlock.getArgument(3) ,declare.getResult(), TRUESSA.getResult());

        // int finished =False;
        auto declare3 = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getI32Type());
        auto assign4 = builder.create<mlir::starplat::AssignmentOp>(builder.getUnknownLoc(), declare3.getResult(), FALSESSA.getResult());

        // fixedPoint until (finished:!modified) 

        auto cond = builder.getStringAttr("NTEQ");

        auto argCondAttr = builder.getArrayAttr({cond});
        llvm::SmallVector<mlir::Value, 2> condArgs = {assign4.getLhs(), assign2.getLhs()};
        auto fixedPoint = builder.create<mlir::starplat::FixedPointUntilOp>(builder.getUnknownLoc(), condArgs, argCondAttr);
        
        // Change the Region.
        auto &loopBlock = fixedPoint.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&loopBlock);

        // node v
        auto vtype = builder.getType<mlir::starplat::NodeType>();
        auto vtypeAttr = ::mlir::TypeAttr::get(vtype);
        auto declare4 = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), builder.getI32Type(), vtypeAttr);
        

        // forall (v in g.nodes().filter(modified == True)) {}
        




    }

    virtual void visitParamlist(const Paramlist *paramlist) override
    {
    }

    virtual void visitFixedpointUntil(const FixedpointUntil *fixedpointuntil) override
    {
    }

    virtual void visitInitialiseAssignmentStmt(const InitialiseAssignmentStmt *initialiseAssignmentStmt)
    {
    }

    virtual void visitMemberAccessAssignment(const MemberAccessAssignment *memberAccessAssignment)
    {
    }

    virtual void visitKeyword(const Keyword *keyword) override
    {
    }

    virtual void visitGraphProperties(const GraphProperties *graphproperties) override
    {
    }

    virtual void visitMethodcall(const Methodcall *methodcall) override
    {
    }

    virtual void visitMemberaccess(const Memberaccess *memberaccess) override
    {
    }

    virtual void visitArglist(const Arglist *arglist) override
    {
    }

    virtual void visitArg(const Arg *arg) override
    {
    }

    virtual void visitStatement(const Statement *statement) override
    {
    }

    virtual void visitStatementlist(const Statementlist *stmtlist) override
    {
    }

    virtual void visitType(const TypeExpr *type) override
    {
    }

    virtual void visitNumber(const Number *number) override
    {
    }

    virtual void visitExpression(const Expression *expr) override
    {
    }

    void print()
    {
        verify(module);
        module.dump();
    }

private:
    int result;
    map<string, int> variables;

    mlir::MLIRContext context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
};