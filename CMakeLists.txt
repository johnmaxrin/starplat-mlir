cmake_minimum_required(VERSION 2.9)
project(Avial)

set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Define Flex and Bison files
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

set(PARSER_DIR "${CMAKE_SOURCE_DIR}/parser")

# Flex
set(LEX_FILE ${PARSER_DIR}/avial.l)
set(LEX_OUTPUT ${PARSER_DIR}/lex.yy.cc)
FLEX_TARGET(lexer ${LEX_FILE} ${LEX_OUTPUT})

# Bison
set(YACC_FILE ${PARSER_DIR}/avial.y)
set(YACC_OUTPUT ${PARSER_DIR}/avial.tab.cc)
set(YACC_HEADER ${PARSER_DIR}/avial.tab.h)
BISON_TARGET(parser ${YACC_FILE} ${YACC_OUTPUT} DEFINES_FILE ${YACC_HEADER})

# Include the directory with the generated header
include_directories(
            ${CMAKE_SOURCE_DIR}/ast 
            ${CMAKE_SOURCE_DIR}/codegen 
            ${PARSER_DIR})

# Add executable
add_executable(app main.cc ${FLEX_lexer_OUTPUTS} ${BISON_parser_OUTPUTS})


# Link libraries (optional if Flex/Bison need any)
target_link_libraries(app ${FLEX_LIBRARIES} ${BISON_LIBRARIES})