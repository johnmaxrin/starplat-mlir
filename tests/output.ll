; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Graph = type { i64 }

declare ptr @malloc(i64)

define ptr @assignment(%Graph %0, i64 %1) {
  %3 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (i1, ptr null, i64 1) to i64))
  %4 = insertvalue { ptr, ptr, i64 } undef, ptr %3, 0
  %5 = insertvalue { ptr, ptr, i64 } %4, ptr %3, 1
  %6 = insertvalue { ptr, ptr, i64 } %5, i64 0, 2
  %7 = extractvalue { ptr, ptr, i64 } %6, 1
  store i1 true, ptr %7, align 1
  %8 = extractvalue { ptr, ptr, i64 } %6, 1
  %9 = ptrtoint ptr %8 to i64
  %10 = inttoptr i64 %9 to ptr
  ret ptr %10
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
