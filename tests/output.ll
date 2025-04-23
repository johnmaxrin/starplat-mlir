; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%Graph = type { i32 }

define i32 @Declare(%Graph %0) {
  %2 = extractvalue %Graph %0, 0
  ret i32 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
