module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @assignment(%arg0: !llvm.struct<"Graph", (i64)>, %arg1: i64) -> !llvm.ptr {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.zero : !llvm.ptr
    %2 = llvm.getelementptr %1[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %3 = llvm.ptrtoint %2 : !llvm.ptr to i64
    %4 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.insertvalue %4, %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.insertvalue %8, %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    %10 = llvm.mlir.constant(true) : i1
    %11 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %10, %11 : i1, !llvm.ptr
    %12 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64)> 
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.inttoptr %13 : i64 to !llvm.ptr
    llvm.return %14 : !llvm.ptr
  }
}

