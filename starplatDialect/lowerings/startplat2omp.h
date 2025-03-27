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
                    auto const1 = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
                    auto const0 = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
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

                                    auto alloc = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(), LLVM::LLVMPointerType::get(funcOp->getContext()), rewriter.getI32Type(), const1);

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
                            auto inf = rewriter.getI32IntegerAttr(2147483647);
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), inf);
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
                            auto falseAttr = rewriter.getI32IntegerAttr(0);
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), falseAttr);
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
                            auto falseAttr = rewriter.getI32IntegerAttr(1);
                            auto alloc = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getI32Type(), falseAttr);
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
    mlir::LLVM::LLVMStructType nodeType = createNodeStruct(rewriter,context);


    // Define Graph struct body with (Node*, int)
    structType.setBody({nodeType, rewriter->getI32Type()}, /*isPacked=*/false);


    //structType.setBody({rewriter->getI32Type()}, false);

    return structType;
}

mlir::LLVM::LLVMStructType createNodeStruct(mlir::IRRewriter *rewriter, mlir::MLIRContext *context)
{
    auto structType = LLVM::LLVMStructType::getIdentified(context, "Node");
    
    // Create a ptr type
    // auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter->getI32Type());
    
    //structType.setBody({rewriter->getI32Type(), ptr}, false);

    return structType;
}


void attachNodeProp()
{
    // Add Grahp at first Param followed by props :done
    // Add Node * nodes to graph struct
    // Add Void ** props to node struct. 
    // Get element pointer.

}
