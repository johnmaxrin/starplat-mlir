module {
  llvm.func @Declare(%arg0: !llvm.struct<"Graph", (i32)>) -> i32 {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<"Graph", (i32)> 
    llvm.return %0 : i32
  }
}
