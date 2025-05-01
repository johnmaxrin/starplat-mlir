// main.cpp
#include <iostream>
#include <cstdint>
#include <cstdlib>

struct Graph
{
    int64_t numofnodes;
};



extern "C"
{
    bool* assignment(Graph g, int64_t src);
}; // assuming you link to LLVM IR or a C-style compiled file

int main()
{
    Graph g;
    g.numofnodes = 10;

    bool* prop = assignment(g,1);


   


    std::cout<<"Arr "<< prop[0] <<"\n";
    



    return 0;
}