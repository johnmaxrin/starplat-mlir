# StarPlat-MLIR

**StarPlat-MLIR** is a compiler infrastructure project that implements the StarPlat DSL using [MLIR (Multi-Level Intermediate Representation)](https://mlir.llvm.org/). It provides a custom dialect for StarPlat and includes various analysis and transformation passes to optimize graph workloads for heterogeneous execution.

## ✨ Features

- **StarPlat Dialect**: A custom MLIR dialect designed to represent high-level graph computations in the StarPlat DSL.
- **Lowering to LLVM**: Supports lowering StarPlat IR all the way down to LLVM dialect, enabling native code generation.
- **Graph-specific Optimizations**:
  - **Edge-to-Vertex Conversion**
  - **Vertex-to-Edge Conversion**
  - **Push-to-Pull Conversion**
  - **Pull-to-Push Conversion**

 

