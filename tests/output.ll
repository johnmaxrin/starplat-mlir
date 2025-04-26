; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Graph = type { i64 }

declare ptr @malloc(i64)

define ptr @attachNodeProperty(%Graph %0) {
  %2 = extractvalue %Graph %0, 0
  %3 = getelementptr i1, ptr null, i64 %2
  %4 = ptrtoint ptr %3 to i64
  %5 = call ptr @malloc(i64 %4)
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %5, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 0, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 1, 4, 0
  %11 = extractvalue %Graph %0, 0
  br label %12

12:                                               ; preds = %15, %1
  %13 = phi i64 [ %18, %15 ], [ 0, %1 ]
  %14 = icmp slt i64 %13, %11
  br i1 %14, label %15, label %19

15:                                               ; preds = %12
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %17 = getelementptr i1, ptr %16, i64 %13
  store i1 true, ptr %17, align 1
  %18 = add i64 %13, 1
  br label %12

19:                                               ; preds = %12
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %21 = ptrtoint ptr %20 to i64
  %22 = inttoptr i64 %21 to ptr
  ret ptr %22
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
