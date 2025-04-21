#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"

// TODO: 
// 1. I have simply added true for filter in forall. Change it! 

bool operationContainsOldValues(mlir::Operation *op, const llvm::DenseMap<mlir::Value, mlir::Value> &operandMapping);

namespace mlir
{
    namespace starplat
    {
#define GEN_PASS_DEF_VERTEXTOEDGE
#include "tblgen2/Passes.h.inc"

        struct VertexToEdge : public mlir::starplat::impl::VertexToEdgeBase<VertexToEdge>
        {
            using VertexToEdgeBase::VertexToEdgeBase;

            /*
                TODOs
                1. Add checks for whether we are traversing all the nodes in a gprah g.
                2. Use symbol table on the first walk and use the data in it for transforming.
            */

            void runOnOperation() override
            {
                auto mod = getOperation();
                bool isPossible = false;
                llvm::DenseMap<mlir::Value, mlir::Value> operandMapping;

                mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                                                     {
                                                         if (llvm::isa<mlir::starplat::ForAllOp>(op))
                                                         {
                                                             auto attr = op->getAttr("loopattributes");
                                                             if (attr)
                                                             {
                                                                 if (auto arrAttr = attr.dyn_cast<mlir::ArrayAttr>())
                                                                 {
                                                                     for (auto attrElm : arrAttr)
                                                                     {
                                                                         if (auto strAttr = attrElm.dyn_cast<mlir::StringAttr>())
                                                                         {
                                                                             if (strAttr.getValue() == "nodes")
                                                                             {
                                                                                 llvm::outs() << "We have a forall loop which iterates over all the nodes.\n";
                                                                                 op->walk<mlir::WalkOrder::PreOrder>([&](mlir::starplat::ForAllOp op)
                                                                                                                     {
                                                                                                                         auto nbrAttr = op->getAttr("loopattributes");
                                                                                                                         if (nbrAttr)
                                                                                                                         {
                                                                                                                             if (auto nbrArrAttr = nbrAttr.dyn_cast<mlir::ArrayAttr>())
                                                                                                                             {
                                                                                                                                 for (auto nbrAttrElm : nbrArrAttr)

                                                                                                                                     if (auto nbrStrAttr = nbrAttrElm.dyn_cast<mlir::StringAttr>())
                                                                                                                                     {
                                                                                                                                         if (nbrStrAttr.getValue() == "neighbours")
                                                                                                                                         {
                                                                                                                                             llvm::outs() << "We have another forall loop which iterates over all neighbours of a node.\n";
                                                                                                                                             llvm::outs() << "Vertex to edge transformation is feasible.\n";
                                                                                                                                             isPossible = true;
                                                                                                                                         }
                                                                                                                                     }
                                                                                                                             }
                                                                                                                         }
                                                                                                                     });
                                                                             }
                                                                         }
                                                                     }
                                                                 }
                                                             }
                                                         } });

                // To do the transformation
                if (isPossible)
                {
                    mod->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                                                         {
                        if(mlir::isa<mlir::starplat::ForAllOp>(op))
                        {
                            auto attr = op->getAttr("loopattributes");
                            if(attr)
                            {
                                if(auto arrAttr = attr.dyn_cast<mlir::ArrayAttr>())
                                {
                                    mlir::SmallVector<mlir::Attribute> attributesOuter(arrAttr.begin(), arrAttr.end());
                                    for(size_t i = 0; i < attributesOuter.size(); i++)
                                    {
                                        if(auto strAttr = attributesOuter[i].dyn_cast<mlir::StringAttr>())
                                        {
                                            if(strAttr.getValue() == "nodes")
                                            {



                                                OpBuilder builder(op);
                                                builder.setInsertionPoint(op);
                                                mlir::SmallVector<mlir::Value> operands = op->getOperands();
                                                ArrayAttr loopAttributes = op->getAttrOfType<mlir::ArrayAttr>("loopattributes");
                                                mlir::SmallVector<mlir::Attribute> attributes(loopAttributes.begin(), loopAttributes.end());
                                                
                                                attributes[i] = builder.getStringAttr("edges");
                                                char *edgeVarName = "v_e";

                                                mlir::Value graph = NULL;
                                                auto edgeVar = builder.create<mlir::starplat::DeclareOp>(builder.getUnknownLoc(), mlir::starplat::EdgeType::get(builder.getContext()), builder.getStringAttr(edgeVarName), builder.getStringAttr("public"), graph);
                                                mlir::Value oldV = operands[1];
                                                mlir::Value oldNbr;
                                                operands[1] = edgeVar.getResult(); // Second variable is always the loop variable.
                                                
                    
                                                auto edgeForall = builder.create<mlir::starplat::ForAllOp>(builder.getUnknownLoc(),operands,builder.getArrayAttr(attributes), builder.getBoolAttr(0), builder.getStringAttr("sampleLoop"));

                                                auto &loopBlock = edgeForall.getBody().emplaceBlock();
                                                builder.setInsertionPointToStart(&loopBlock);

                                                auto src = builder.create<mlir::starplat::GetEdgeOp>(builder.getUnknownLoc(), builder.getI32Type(),  edgeVar.getResult(),edgeVar.getResult(),edgeVar.getResult());
                                                auto dst = builder.create<mlir::starplat::GetEdgeOp>(builder.getUnknownLoc(), builder.getI32Type(), edgeVar.getResult(),edgeVar.getResult() ,edgeVar.getResult());
                                                


                                                op->getRegion(0).walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                                                    {
                                                        if(mlir::isa<mlir::starplat::DeclareOp>(op))
                                                        {
                                                            // save the nbr opereand. Change this! 
                                                            oldNbr = op->getResult(0);

                                                        }

                                                        else if(mlir::isa<mlir::starplat::ForAllOp>(op))
                                                        {
                                                            op->getRegion(0).walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op)
                                                        {
                                                           for(mlir::Value opr : op->getOperands())
                                                           {
                                                            if(opr == oldV)
                                                            {

                                                                if(mlir::isa<mlir::starplat::GetNodePropertyOp>(op))
                                                                {   
                                                                    auto propertyAttr = op->getAttrOfType<StringAttr>("property");
                                                                    auto useSrc = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), src->getResult(0), op->getResult(0), propertyAttr);
                                                                    operandMapping[op->getResult(0)]= useSrc->getResult(0);
                                                                
                                                                }


                                                                llvm::outs() << "\n\n";

                                                                break;
                                                            }

                                                            else if(opr == oldNbr)
                                                            {
                                                                if(mlir::isa<mlir::starplat::GetNodePropertyOp>(op))
                                                                {   
                                                                    auto propertyAttr = op->getAttrOfType<StringAttr>("property");
                                                                    auto useDst = builder.create<mlir::starplat::GetNodePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), dst->getResult(0), op->getResult(0), propertyAttr);
                                                                    operandMapping[op->getResult(0)]= useDst->getResult(0);

                                                                }


                                                                llvm::outs() << "\n\n";

                                                                break;
                                                            }

                                                            
                                                           }
                                                           if(mlir::isa<mlir::starplat::GetEdgePropertyOp>(op))
                                                            {
                                                                auto propertyAttr = op->getAttrOfType<StringAttr>("property");
                                                                auto edgeProp = builder.create<mlir::starplat::GetEdgePropertyOp>(builder.getUnknownLoc(), builder.getI32Type(), edgeVar->getResult(0), op->getResult(0), propertyAttr);
                                                                operandMapping[op->getResult(0)]= edgeProp->getResult(0);

                                                            }

                                                            if(mlir::isa<mlir::starplat::AddOp>(op))
                                                            {
                                                                mlir::Value op1 = op->getOperand(0);
                                                                mlir::Value op2 = op->getOperand(1);


                                                                mlir::Value newOp1 = op1;
                                                                mlir::Value newOp2 = op2;

                                                                if(operationContainsOldValues(op,operandMapping))
                                                                {
                                                                    if(operandMapping.contains(op1))
                                                                        newOp1 = operandMapping[op1];
                                                                    
                                                                    if(operandMapping.contains(op2))
                                                                        newOp2 = operandMapping[op2];
                                                                }

                                                                auto addop = builder.create<mlir::starplat::AddOp>(builder.getUnknownLoc(), builder.getI32Type(), newOp1, newOp2);
                                                                operandMapping[op->getResult(0)]= addop->getResult(0);
                                                            }

                                                            if(mlir::isa<mlir::starplat::MinOp>(op))
                                                            {
                                                                mlir::Value op1 = op->getOperand(0);
                                                                mlir::Value op2 = op->getOperand(1);
                                                                mlir::Value op3 = op->getOperand(2);
                                                                mlir::Value op4 = op->getOperand(3);

                                                                mlir::Value newOp1 = op1;
                                                                mlir::Value newOp2 = op2;
                                                                mlir::Value newOp3 = op3;
                                                                mlir::Value newOp4 = op4;

                                                                if(operationContainsOldValues(op,operandMapping))
                                                                {
                                                                    if(operandMapping.contains(op1))
                                                                        newOp1 = operandMapping[op1];
                                                                    
                                                                    if(operandMapping.contains(op2))
                                                                        newOp2 = operandMapping[op2];

                                                                    if(operandMapping.contains(op3))
                                                                        newOp3 = operandMapping[op3];

                                                                    
                                                                    if(operandMapping.contains(op4))
                                                                        newOp4 = operandMapping[op4];
                                                                }
                                                            
                                                                auto minOp = builder.create<mlir::starplat::MinOp>(builder.getUnknownLoc(), builder.getI32Type(),newOp2, newOp1, newOp2, newOp3, newOp4); 
                                                            
                                                            }


                                                           
                                                        });
                                                        }

                                                    });

                                                builder.create<mlir::starplat::endOp>(builder.getUnknownLoc());
                    
                                                op->erase();
                                            }
                                        }
                                    }
                                }
                            }

                            

                        } });
                }

                llvm::outs() << "Finished Vertex to Edge Transform\n";
            }
        };

    }
}

bool operationContainsOldValues(mlir::Operation *op, const llvm::DenseMap<mlir::Value, mlir::Value> &operandMapping)
{
    for (mlir::Value operand : op->getOperands())
    {
        if (operandMapping.count(operand))
        {
            return true;
        }
    }
    return false;
}
