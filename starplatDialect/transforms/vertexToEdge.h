#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "includes/StarPlatDialect.h"



namespace mlir
{
    namespace starplat
    {
#define GEN_PASS_DEF_VERTEXTOEDGE
#include "tblgen2/Passes.h.inc"

        struct VertexToEdge : public mlir::starplat::impl::VertexToEdgeBase<VertexToEdge>
        {
            using VertexToEdgeBase::VertexToEdgeBase;

            void runOnOperation() override
            {
                auto mod = getOperation();

                mod->walk([&](mlir::Operation *op) {

                });

                llvm::outs() << "Vertex to Edge Transform\n";
            }
        };

    }
}
