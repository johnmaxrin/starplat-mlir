// main.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{

    struct Graph
    {
        //int *array;
        int numofnodes;
        //int *buffer;
    };

    int Declare(struct Graph g);
}

int main()
{

    struct Graph g;

    int a[4] = {1,2,3,4};
    int b[4] = {1,2,3,4};


    g.numofnodes = 10;
    //g.buffer = a;
    //g.array = b;
    

    int result = Declare(g);
    printf("Result: %d\n", result);

    return 0;
}
