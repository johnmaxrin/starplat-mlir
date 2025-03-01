#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"
#include "includes/StarPlatOps.h"


#include "../transforms/reachingDef.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"


namespace mlir
{
    namespace starplat
    {
#define  GEN_PASS_DEF_REACHDEF
#include "tblgen2/Passes.h.inc"
    } // namespace starplat
} // namespace mlir

struct ReachDef : public mlir::starplat::impl::ReachDefBase<ReachDef>
{
    using ReachDefBase::ReachDefBase;
    void runOnOperation() override
    {
        auto mod = getOperation();

        llvm::outs() << "Hello\n";
        mod->walk([&](mlir::starplat::DeclareOp declop)
            {
                declop.dump();
                llvm::outs() << "Hello\n";
            }   
        );

    
    }
};