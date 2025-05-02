#include <stdio.h>

// CUDA核函数（在GPU上执行的函数）
__global__ void helloWorld()
{
    // TODO 打印hello world、块ID、线程ID、线程唯一ID
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = bid * blockDim.x + tid;
    printf("hello world from block %d thread %d id %d\n", bid, tid, id);
}

int main()
{
    // 主机代码（CPU执行）
    printf("Hello World from CPU!\n");

    // TODO 启动2个块，每个块4个线程的核函数, 对应一个grid
    grid_size=2, block_size=4;
    helloWorld<<<grid_size, block_size>>>();

    // 等待GPU完成
    cudaDeviceSynchronize();

    return 0;
}