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
    bool* attachNodeProperty(Graph g, int64_t src);
}; // assuming you link to LLVM IR or a C-style compiled file

int main()
{
    Graph g;
    g.numofnodes = 10;

    bool* prop = attachNodeProperty(g,1);


   

    for(int i=0; i<10; ++i)
        std::cout<<"Arr "<< prop[i] <<"\n";
    



    return 0;
}
