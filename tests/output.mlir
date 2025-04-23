module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @attachNodeProperty(%arg0: !llvm.struct<"Graph", (i32)>) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<"Graph", (i32)> 
    %1 = llvm.sext %0 : i32 to i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.zero : !llvm.ptr
    %4 = llvm.getelementptr %3[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %5 = llvm.ptrtoint %4 : !llvm.ptr to i64
    %6 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.insertvalue %10, %9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %1, %11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %2, %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.mlir.constant(false) : i1
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.extractvalue %arg0[0] : !llvm.struct<"Graph", (i32)> 
    %17 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%15 : i32)
  ^bb1(%18: i32):  // 2 preds: ^bb0, ^bb2
    %19 = llvm.icmp "slt" %18, %16 : i32
    llvm.cond_br %19, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %20 = llvm.sext %18 : i32 to i64
    %21 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.getelementptr %21[%20] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %14, %22 : i1, !llvm.ptr
    %23 = llvm.add %18, %17 : i32
    llvm.br ^bb1(%23 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return %13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}
