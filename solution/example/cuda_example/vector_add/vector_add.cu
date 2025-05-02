#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // 向量长度

// CUDA核函数：向量加法
__global__ void vectorAdd(int *a, int *b, int *c)
{
    // TODO 1: 计算全局索引（一维网格）
    // 提示：使用blockDim.x, blockIdx.x, threadIdx.x
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保不越界
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *a, *b, *c;         // 主机（host）指针
    int *d_a, *d_b, *d_c;   // 设备（device）指针


    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    // TODO 同理分配主机内存
    c = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    // TODO 同理分配设备全局内存
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    // TODO 同理把数据从主机拷贝到设备
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 配置核函数参数
    int threadsPerBlock = 256;
    // TODO 4: 计算所需的块数量（blocksPerGrid=4）
    int blocksPerGrid = N / threadsPerBlock;

    // 启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // TODO 6: 拷贝结果回主机（cudaMemcpy, DeviceToHost）
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出数组内容
    printf("c:"); for(int i=0; i<N; i++)printf("%d ", c[i]); printf("\n");
    printf("a:"); for(int i=0; i<N; i++)printf("%d ", a[i]); printf("\n");
    printf("b:"); for(int i=0; i<N; i++)printf("%d ", b[i]); printf("\n");

    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            break;
        }
    }
    printf("Test %s!\n", success ? "Passed" : "Failed");

    cudaFree(d_a);
    cudaFree(d_b);
    // TODO 7: 释放设备内存（cudaFree）
    cudaFree(d_c);

    free(a);
    free(b);
    // TODO 释放主机内存
    free(c);

    return 0;
}