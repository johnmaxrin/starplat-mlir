; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Graph = type { i32 }

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @attachNodeProperty(%Graph %0) {
  %2 = extractvalue %Graph %0, 0
  %3 = sext i32 %2 to i64
  %4 = getelementptr i1, ptr null, i64 %3
  %5 = ptrtoint ptr %4 to i64
  %6 = call ptr @malloc(i64 %5)
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %6, 0
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %6, 1
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 0, 2
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %3, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 1, 4, 0
  %12 = extractvalue %Graph %0, 0
  br label %13

13:                                               ; preds = %16, %1
  %14 = phi i32 [ %20, %16 ], [ 0, %1 ]
  %15 = icmp slt i32 %14, %12
  br i1 %15, label %16, label %21

16:                                               ; preds = %13
  %17 = sext i32 %14 to i64
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %19 = getelementptr i1, ptr %18, i64 %17
  store i1 false, ptr %19, align 1
  %20 = add i32 %14, 1
  br label %13

21:                                               ; preds = %13
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %11
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
