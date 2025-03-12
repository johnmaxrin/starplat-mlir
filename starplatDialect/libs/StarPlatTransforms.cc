#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"

#include "../transforms/reachingDef.h"
#include "../lowerings/startplat2omp.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir
{
    namespace starplat
    {
#define GEN_PASS_DEF_REACHDEF

#include "tblgen2/Passes.h.inc"
    } // namespace starplat
} // namespace mlir



struct ReachDef : public mlir::starplat::impl::ReachDefBase<ReachDef>
{
    using ReachDefBase::ReachDefBase;
    void runOnOperation() override
    {
        llvm::SmallVector<mlir::Value> defs;

        auto mod = getOperation();

        mod->walk([&](mlir::starplat::DeclareOp declareOp)
                  { defs.push_back(declareOp->getResult(0)); });

        llvm::outs() << "Total Defs: " << defs.size();

        mod->walk([&](mlir::Operation *op)
                  {
            for(auto operand : op->getOperands())
            {
                auto it = std::find(defs.begin(), defs.end(), operand);
        if (it != defs.end()) 
        {
            defs.erase(it); // Remove from defs if used
        }
            } });

        llvm::outs() << "Total defs unused: " << defs.size() << "\n";

        for(auto udef : defs)
            udef.dump();
    }
};

