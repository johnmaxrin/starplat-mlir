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
    bool* attachNodeProperty(Graph g);
}; // assuming you link to LLVM IR or a C-style compiled file

int main()
{
    Graph g;
    g.numofnodes = 10;

    bool* prop = attachNodeProperty(g);


   

    for(int i=0; i<10; ++i)
        std::cout<<"Arr "<< prop[i] <<"\n";
    



    return 0;
}
