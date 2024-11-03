# Find Flex and Bison
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Define output directory for generated files in the build directory
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/parser")
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Define paths for Flex and Bison outputs
set(LEX_FILE "${CMAKE_SOURCE_DIR}/parser/avial.l")
set(LEX_OUTPUT "${GENERATED_DIR}/lex.yy.cc")
FLEX_TARGET(lexer ${LEX_FILE} ${LEX_OUTPUT})

set(YACC_FILE "${CMAKE_SOURCE_DIR}/parser/avial.y")
set(YACC_OUTPUT "${GENERATED_DIR}/avial.tab.cc")
set(YACC_HEADER "${GENERATED_DIR}/avial.tab.h")
BISON_TARGET(parser ${YACC_FILE} ${YACC_OUTPUT} DEFINES_FILE ${YACC_HEADER})

# Add library target to compile parser files
add_library(parser_lib ${FLEX_lexer_OUTPUTS} ${BISON_parser_OUTPUTS})

# Export generated files for use in the main CMakeLists.txt
set(FLEX_lexer_OUTPUTS ${LEX_OUTPUT} PARENT_SCOPE)
set(BISON_parser_OUTPUTS ${YACC_OUTPUT} PARENT_SCOPE)
