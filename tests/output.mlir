module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @attachNodeProperty(%arg0: !llvm.struct<"Graph", (i64)>) -> !llvm.ptr {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<"Graph", (i64)> 
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.constant(0 : index) : i64
    %10 = llvm.insertvalue %9, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %0, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %1, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(true) : i1
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.extractvalue %arg0[0] : !llvm.struct<"Graph", (i64)> 
    %16 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%17: i64):  // 2 preds: ^bb0, ^bb2
    %18 = llvm.icmp "slt" %17, %15 : i64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %19 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.getelementptr %19[%17] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %13, %20 : i1, !llvm.ptr
    %21 = llvm.add %17, %16 : i64
    llvm.br ^bb1(%21 : i64)
  ^bb3:  // pred: ^bb1
    %22 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    llvm.return %24 : !llvm.ptr
  }
}
