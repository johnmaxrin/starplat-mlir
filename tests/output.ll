; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Graph = type { i64 }

declare ptr @malloc(i64)

define ptr @attachNodeProperty(%Graph %0, i64 %1) {
  %3 = extractvalue %Graph %0, 0
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

13:                                               ; preds = %16, %2
  %14 = phi i64 [ %19, %16 ], [ 0, %2 ]
  %15 = icmp slt i64 %14, %12
  br i1 %15, label %16, label %20

16:                                               ; preds = %13
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %18 = getelementptr i1, ptr %17, i64 %14
  store i1 false, ptr %18, align 1
  %19 = add i64 %14, 1
  br label %13

20:                                               ; preds = %13
  %21 = extractvalue %Graph %0, 0
  %22 = icmp sgt i64 %21, %1
  br i1 %22, label %23, label %26

23:                                               ; preds = %20
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %25 = getelementptr i1, ptr %24, i64 %1
  store i1 true, ptr %25, align 1
  br label %26

26:                                               ; preds = %23, %20
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %28 = ptrtoint ptr %27 to i64
  %29 = inttoptr i64 %28 to ptr
  ret ptr %29
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
