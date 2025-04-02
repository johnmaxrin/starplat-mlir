#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// mlir::Operation *generateGraphStruct(mlir::MLIRContext *context)
// {
//     auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");

//     return;
// }

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context);
void lowerAttachNodePropOp(mlir::Operation *attachNodePropOp, mlir::IRRewriter *rewriter, mlir::Operation *numOfNodes);
void lowerSetNodePropOp(mlir::Operation *setNodePropOp, mlir::IRRewriter *rewriter);
void lowerFixedPoint(mlir::Operation *fixedPointOp, mlir::IRRewriter *rewriter ,mlir::Operation *funcOp, mlir::Operation *moduleOp, mlir::Operation *numOfNodes);
LLVM::LLVMFuncOp createLLVMReductionFunction(mlir::Operation *modOp, mlir::IRRewriter *rewriter, mlir::Block *prevPoint);

namespace mlir
{
    namespace starplat
    {
#define GEN_PASS_DEF_CONVERTSTARTPLATIRTOOMPPASS
#include "tblgen2/Passes.h.inc"

        struct ConvertStartPlatIRToOMPPass : public mlir::starplat::impl::ConvertStartPlatIRToOMPPassBase<ConvertStartPlatIRToOMPPass>
        {
            using ConvertStartPlatIRToOMPPassBase::ConvertStartPlatIRToOMPPassBase;

            void runOnOperation() override
            {
                auto mod = getOperation();
                mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::starplat::FuncOp funcOp)
                {
                    funcOp->getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
                    mlir::IRRewriter rewriter(funcOp->getContext());
                    mlir::OpBuilder::InsertionGuard guard(rewriter);

                    auto &entryBlock = funcOp.getBody().getBlocks().front();
                    rewriter.setInsertionPointToStart(&entryBlock);
                    auto const1 = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(1));
                    auto const0 = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI8Type(), rewriter.getI8IntegerAttr(0));
                    mlir::Operation *numofNodes = nullptr;
                    // CSR For Vertex based.
                    // COO for Edge based.

                    // Check arg ops and declare structs. Eg Graph, Nodes etc
                    llvm::SmallVector<mlir::Operation *, 4> toErase;
                    
                    funcOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                    {
                    
                     
                    if(llvm::isa<mlir::starplat::ArgOp>(op))
                    {
                        auto arg = llvm::cast<mlir::starplat::ArgOp>(op);
                    if (auto typeAttr = arg->getAttrOfType<mlir::TypeAttr>("type")) {
                        mlir::Type argType = typeAttr.getValue();
                    
                        // Check if argType is the specific Starplat type
                        if (argType.isa<mlir::starplat::GraphType>()) {

                            auto graphStruct = createGraphStruct(&rewriter, funcOp->getContext());
                            auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), graphStruct,const1);
                            auto graphStructVal = rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(),createGraphStruct(&rewriter, rewriter.getContext()), alloc);

                            numofNodes = rewriter.create<LLVM::ExtractValueOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), graphStructVal,rewriter.getDenseI64ArrayAttr({1}));
                            
                            if(arg->hasAttr("sym_name")){
                                auto symNameAttr = arg->getAttrOfType<mlir::StringAttr>("sym_name");
                                if(symNameAttr){
                                    alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                    alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                        }
                                    }
                                        arg->replaceAllUsesWith(alloc);
                                        toErase.push_back(arg);
                                }

                                else if(argType.isa<mlir::starplat::NodeType>()) {
                                    auto nodeStruct = createNodeStruct(&rewriter, funcOp->getContext());
                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), nodeStruct, const1);
                                    
                                    if (arg->hasAttr("sym_name")) {
                                        auto symNameAttr = arg->getAttrOfType<mlir::StringAttr>("sym_name");
                                        if (symNameAttr) {
                                            alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                            alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                        }
                                    }

                                        arg->replaceAllUsesWith(alloc);
                                        toErase.push_back(arg);
                                    
                                }

                                else if(argType.isa<mlir::starplat::PropNodeType>()){

                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), numofNodes->getResult(0));

                                    if (arg->hasAttr("sym_name")) {
                                        auto symNameAttr = arg->getAttrOfType<mlir::StringAttr>("sym_name");
                                        if (symNameAttr) {
                                            alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                            alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                        }
                                    }
                                    arg->replaceAllUsesWith(alloc);
                                    toErase.push_back(arg);

                                }


                            } 
                        }


                    else if(llvm::isa<mlir::starplat::DeclareOp>(op))
                    {
                        auto declOp = llvm::cast<mlir::starplat::DeclareOp>(op);
                        if (auto typeAttr = declOp->getAttrOfType<mlir::TypeAttr>("type"))
                        {
                            mlir::Type argType = typeAttr.getValue();
                            if (argType.isa<mlir::starplat::PropNodeType>())
                            {
                                auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), numofNodes->getResult(0));

                                if (declOp->hasAttr("sym_name"))
                                {
                                    auto symNameAttr = declOp->getAttrOfType<mlir::StringAttr>("sym_name");
                                    if (symNameAttr)
                                    {
                                        alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                        alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                    }
                                }
                                declOp->replaceAllUsesWith(alloc);
                                toErase.push_back(declOp);
                            }

                            else if(argType.isa<mlir::IntegerType>())
                            {
                                auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), const1);

                                if (declOp->hasAttr("sym_name"))
                                {
                                    auto symNameAttr = declOp->getAttrOfType<mlir::StringAttr>("sym_name");
                                    if (symNameAttr)
                                    {
                                        alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                        alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                    }
                                }
                                declOp->replaceAllUsesWith(alloc);
                                toErase.push_back(declOp);
                            }
                        }
                    }

                    else if(llvm::isa<mlir::starplat::ConstOp>(op))
                    {

                            auto constOp = llvm::cast<mlir::starplat::ConstOp>(op);
                            if (auto typeAttr = constOp->getAttrOfType<mlir::StringAttr>("value")) {
                            
                            string constType = typeAttr.str();

                            if(constType == "INF")
                            {
                            auto inf = rewriter.getI8IntegerAttr(100); // Will think about this later [TODO!]
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI8Type(), inf);
                            // auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), inf);

                            if (constOp->hasAttr("sym_name")) {
                                auto symNameAttr = constOp->getAttrOfType<mlir::StringAttr>("sym_name");
                                if (symNameAttr) {
                                    alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                    alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                }
                            }
                            constOp->replaceAllUsesWith(alloc);
                            toErase.push_back(constOp);
                            }

                            else if(constType == "False")
                            {
                            auto falseAttr = rewriter.getI8IntegerAttr(0);
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI8Type(), falseAttr);
                            // auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), inf);

                            if (constOp->hasAttr("sym_name")) {
                                auto symNameAttr = constOp->getAttrOfType<mlir::StringAttr>("sym_name");
                                if (symNameAttr) {
                                    alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                    alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                }
                            }
                            constOp->replaceAllUsesWith(alloc);
                            toErase.push_back(constOp);   
                            }

                            else if(constType == "True")
                            {
                            auto falseAttr = rewriter.getI8IntegerAttr(1);
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI8Type(), falseAttr);
                            // auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), inf);

                            if (constOp->hasAttr("sym_name")) {
                                auto symNameAttr = constOp->getAttrOfType<mlir::StringAttr>("sym_name");
                                if (symNameAttr) {
                                    alloc->setAttr("sym_name", rewriter.getStringAttr(symNameAttr.getValue()));
                                    alloc->setAttr("sym_visibility", rewriter.getStringAttr("nested"));
                                }
                            }
                            constOp->replaceAllUsesWith(alloc);
                            toErase.push_back(constOp);   
                            }

                            else if(constType == "0")
                            {
                                constOp->replaceAllUsesWith(const0);
                                toErase.push_back(constOp);
                            }
                            
                        } 
                    }


                    else if(llvm::isa<mlir::starplat::AssignmentOp>(op))
                    {

                        auto assignOp = llvm::cast<mlir::starplat::AssignmentOp>(op);
                        mlir::Value value = assignOp->getOperand(0);
                        mlir::Value ptr = assignOp->getOperand(1);

                        auto assign = rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), ptr, value);
                        
                        //assignOp->replaceAllUsesWith(assign);
                        toErase.push_back(assignOp);


                    }

                    else if(llvm::isa<mlir::starplat::AttachNodePropertyOp>(op))
                        lowerAttachNodePropOp(op,&rewriter, numofNodes);

                    else if(llvm::isa<mlir::starplat::SetNodePropertyOp>(op))
                        lowerSetNodePropOp(op, &rewriter);

                    else if(llvm::isa<mlir::starplat::FixedPointUntilOp>(op))
                        lowerFixedPoint(op, &rewriter, funcOp, mod, numofNodes);

                    });

           

                for (mlir::Operation *op : toErase)
                    op->erase();


                mod->dump();
            });

            
                
            


            }

            

        };
    }
}

mlir::LLVM::LLVMStructType createGraphStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{

    auto structType = LLVM::LLVMStructType::getIdentified(context, "Graph");
    // Create Node struct type

    auto ptrType = LLVM::LLVMPointerType::get(context);
    // Define Graph struct body with (Node*, int)
    structType.setBody({ptrType, rewriter->getI32Type()}, /*isPacked=*/false);


    //structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Node");
    
    // Create a ptr type
    // auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter->getI32Type());
    
    structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}


void lowerAttachNodePropOp(mlir::Operation *attachNodePropOp, mlir::IRRewriter *rewriter, mlir::Operation *numOfNodes)
{
    // So, we have graph[0] and numofnodes[1]
    // Get Numofnodes from graph
    // use it to allocate memory for each props

    // Use numOfNodes to initialize values.
    // Loop through all the operands. 
    
    auto numOfOperands = attachNodePropOp->getNumOperands(); 
    auto numNodesI64 = rewriter->create<LLVM::ZExtOp>(
        rewriter->getUnknownLoc(),
        rewriter->getI64Type(),  // Convert to i64
        numOfNodes->getResult(0)
    );

    auto elementSize = rewriter->create<LLVM::ConstantOp>(
        rewriter->getUnknownLoc(), 
        rewriter->getI64Type(), 
        rewriter->getI64IntegerAttr(4) // Size of i32 = 4 bytes
    );

    auto totalSize = rewriter->create<LLVM::MulOp>(rewriter->getUnknownLoc(), numNodesI64, elementSize);

    for(int i=1; i<numOfOperands; i+=2)
        rewriter->create<LLVM::MemsetOp>(rewriter->getUnknownLoc(), attachNodePropOp->getOperand(i), attachNodePropOp->getOperand(i+1), totalSize, false);


    attachNodePropOp->erase();
   

}


void lowerSetNodePropOp(mlir::Operation *setNodePropOp, mlir::IRRewriter *rewriter)
{
    auto node = setNodePropOp->getOperand(0);
    auto prop = setNodePropOp->getOperand(1);
    auto value = setNodePropOp->getOperand(2);

    // Extract the node value. 
    auto loadnode = rewriter->create<LLVM::LoadOp>(rewriter->getUnknownLoc(), createNodeStruct(rewriter, rewriter->getContext()), node);
    auto nodeVal = rewriter->create<LLVM::ExtractValueOp>(rewriter->getUnknownLoc(), rewriter->getI32Type(), loadnode, rewriter->getDenseI64ArrayAttr({0}));
    auto assign = rewriter->create<LLVM::StoreOp>(rewriter->getUnknownLoc(), value, prop);

    setNodePropOp->erase();    
}

void lowerFixedPoint(mlir::Operation *fixedPointOp, mlir::IRRewriter *rewriter ,mlir::Operation *funcOp, mlir::Operation *moduleOp, mlir::Operation *numOfNodes)
{
    // Get the coditionals predicate operands
    // Create 3 blocks. 
    // loopCond:
    // loopBody:
    // loopExit:


    mlir::Attribute attr = fixedPointOp->getAttr("terminationCondition");

    mlir::Value lhs = fixedPointOp->getOperands()[0];
    mlir::Value rhs = fixedPointOp->getOperands()[1];

    
    mlir::Region &region = funcOp->getRegion(0);
    mlir::Block *loopCond = new mlir::Block();
    region.push_back(loopCond);

    mlir::Block *loopBody = new mlir::Block();
    region.push_back(loopBody);

    mlir::Block *loopExit = new mlir::Block();
    region.push_back(loopExit);
    
    rewriter->create<LLVM::BrOp>(rewriter->getUnknownLoc(), loopCond); 
    rewriter->setInsertionPointToStart(loopCond);

    auto const1 = rewriter->create<LLVM::ConstantOp>(rewriter->getUnknownLoc(), rewriter->getI1Type(), rewriter->getBoolAttr(true));

    // Check the  termination Condition
    auto terminationStr = attr.dyn_cast<mlir::ArrayAttr>()[0].dyn_cast<mlir::StringAttr>();
    if( terminationStr.getValue() == "NOT")
    {
        //rewriter->create<LLVM::OrOp>();

        auto reduceFunc = createLLVMReductionFunction(funcOp, rewriter, loopCond);
        auto redVal = rewriter->create<LLVM::CallOp>(rewriter->getUnknownLoc(), reduceFunc, ArrayRef<mlir::Value>{lhs, rhs});
        auto cond = rewriter->create<LLVM::ICmpOp>(rewriter->getUnknownLoc(), LLVM::ICmpPredicate::ne, redVal->getResult(0), const1);
        rewriter->create<LLVM::CondBrOp>(rewriter->getUnknownLoc(), cond, loopBody, loopExit);
    }

    // Loop Body
    rewriter->setInsertionPointToStart(loopBody);
    // Parse the remaning things here. 
    fixedPointOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
    {

        if(llvm::isa<mlir::starplat::ForAllOp>(op))
        {
            llvm::outs() << "Control For All \n";
        }
    });

    // Loop Exit
    rewriter->setInsertionPointToStart(loopExit);



    
}


LLVM::LLVMFuncOp createLLVMReductionFunction(mlir::Operation *modOp, mlir::IRRewriter *rewriter, mlir::Block *prevPoint) {

    // Think about the return type after --> This could be a potential bug! 

    auto context = modOp->getContext();


    // Set up function type: (i32, i32) -> i32
    Type i32Type = IntegerType::get(context, 32);
    Type ptrType = LLVM::LLVMPointerType::get(context);
    Type i1Type =  IntegerType::get(context, 1);

    LLVM::LLVMFunctionType funcType = LLVM::LLVMFunctionType::get(i1Type,  {ptrType, i32Type}, false);

    rewriter->setInsertionPoint(modOp);
    // Create an LLVM-style function
    LLVM::LLVMFuncOp func = rewriter->create<LLVM::LLVMFuncOp>(
        rewriter->getUnknownLoc(), "reduceOr", funcType);

    // Add an entry block
    auto *entryBlock = rewriter->createBlock(&func.getBody());
    rewriter->setInsertionPointToStart(entryBlock);
    
    ArrayRef<Type> argTypes = funcType.getParams();
    for (Type argType : argTypes) {
        entryBlock->addArgument(argType, rewriter->getUnknownLoc());
    }

    auto const0 = rewriter->create<LLVM::ConstantOp>(rewriter->getUnknownLoc(), rewriter->getI32Type(), rewriter->getI8IntegerAttr(0));
    auto const1 = rewriter->create<LLVM::ConstantOp>(rewriter->getUnknownLoc(), rewriter->getI32Type(), rewriter->getI8IntegerAttr(1));
    auto constFalse = rewriter->create<LLVM::ConstantOp>(rewriter->getUnknownLoc(), i1Type, rewriter->getBoolAttr(0));
    
    auto index = rewriter->create<LLVM::AllocaOp>(rewriter->getUnknownLoc(), ptrType,i32Type, const1);
    rewriter->create<LLVM::StoreOp>(rewriter->getUnknownLoc(), const0, index);

    auto resultPtr = rewriter->create<LLVM::AllocaOp>(rewriter->getUnknownLoc(), ptrType, i1Type,const1);
    rewriter->create<LLVM::StoreOp>(rewriter->getUnknownLoc(), constFalse, resultPtr);

    auto n = func.getArgument(1); // i32, total numbre of nodes. 
    auto ptrArray = func.getArgument(0); // Array pointer. 

    

    auto loopCond = func.addBlock();
    auto loopBody = func.addBlock();
    auto loopExit = func.addBlock();

    auto brOp = rewriter->create<LLVM::BrOp>(rewriter->getUnknownLoc(), loopCond);
    
    // Loop Cond
    rewriter->setInsertionPointToStart(loopCond);
    auto i = rewriter->create<LLVM::LoadOp>(rewriter->getUnknownLoc(), i32Type, index);
    auto cond = rewriter->create<LLVM::ICmpOp>(rewriter->getUnknownLoc(), LLVM::ICmpPredicate::slt, i, n);
    rewriter->create<LLVM::CondBrOp>(rewriter->getUnknownLoc(), cond, loopBody, loopExit);

    // Loop Body
    rewriter->setInsertionPointToStart(loopBody);
    // OR 
    auto eoiptr = rewriter->create<LLVM::GEPOp>(rewriter->getUnknownLoc(),ptrType, i1Type, ptrArray, ArrayRef<Value>{
        rewriter->create<LLVM::ConstantOp>(rewriter->getUnknownLoc(), rewriter->getI32Type(), 0),  // First index (base)
    }); 

    auto eoi = rewriter->create<LLVM::LoadOp>(rewriter->getUnknownLoc(), i1Type, eoiptr);

    auto result = rewriter->create<LLVM::LoadOp>(rewriter->getUnknownLoc(), i1Type, resultPtr);
    auto orVal = rewriter->create<LLVM::OrOp>(rewriter->getUnknownLoc(),i1Type, eoi, result);
    rewriter->create<LLVM::StoreOp>(rewriter->getUnknownLoc(), orVal, resultPtr);


    // Increment index
    
    auto newi = rewriter->create<LLVM::AddOp>(rewriter->getUnknownLoc(), i32Type, i, const1);
    rewriter->create<LLVM::StoreOp>(rewriter->getUnknownLoc(), newi, index);
    rewriter->create<LLVM::BrOp>(rewriter->getUnknownLoc(), loopCond);



    // Return the result
    rewriter->setInsertionPointToStart(loopExit);
    auto resultFin = rewriter->create<LLVM::LoadOp>(rewriter->getUnknownLoc(), i1Type, resultPtr);
    rewriter->create<LLVM::ReturnOp>(rewriter->getUnknownLoc(), resultFin);
    rewriter->setInsertionPointToEnd(prevPoint);


    return func;
}



