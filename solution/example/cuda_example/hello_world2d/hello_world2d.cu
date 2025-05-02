#include <stdio.h>

__global__ void helloWorld2D()
{
    // TODO 计算block_id
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    // TODO 计算thread_id
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    // TODO 计算unique_id
    int threads_per_block = blockDim.x * blockDim.y;
    int unique_id = block_id * threads_per_block + thread_id;

    printf("Hello World from block (%d, %d), thread (%d, %d) | "
           "block_id=%d, thread_id=%d, unique_id=%d\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
           block_id, thread_id, unique_id);
}

int main()
{
    printf("Hello World from CPU!\n");

    dim3 grid_size(3, 2);
    dim3 block_size(4, 2);

    // 启动核函数
    helloWorld2D<<<grid_size, block_size>>>();

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    return 0;
}