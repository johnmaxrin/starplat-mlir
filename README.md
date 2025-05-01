# Starplat MLIR
Starplat-MLIR is an MLIR-based compiler infrastructure for the Starplat domain-specific language (DSL), designed to support high-performance graph analytics and heterogeneous compilation across CPUs, GPUs, and FPGAs. This repository contains custom dialects, IR transformations, and backend lowering pipelines for optimizing and generating parallel code from Starplat programs.

### How to compile?
```bash
chmod +x compile.sh
./compile.sh
```

### How to run?

```bash
.build/app tests/test.avl > ast.avl
```
