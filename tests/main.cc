// main.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{

    struct Graph
    {
        // int *array;
        int numofnodes;
        // int *buffer;
    };

    struct NodeProperty
    {
        void *ptr1;
        void *ptr2;
        int64_t dummy;
        int64_t shape[1];
        int64_t strides[1];
    };

    struct NodeProperty attachNodeProperty(struct Graph g);
}

int main()
{

    struct Graph g;

    int a[4] = {1, 2, 3, 4};
    int b[4] = {1, 2, 3, 4};

    g.numofnodes = 10;
    // g.buffer = a;
    // g.array = b;

    struct NodeProperty prop = attachNodeProperty(g);

    printf("ptr0: %p\n", prop.ptr1);
    printf("ptr1: %p\n", prop.ptr2);
    printf("stride: %lu\n", prop.strides[0]);

    uint8_t *data = (uint8_t *)prop.ptr1;

    for (int i = 0; i < 10; ++i)
        printf("Returned ptr1: %d\n", data[i]);

    return 0;
}
