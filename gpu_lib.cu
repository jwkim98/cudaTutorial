#include "gpu_lib.cuh"

namespace cuda_lib{

    cudaMatrix::MatMulAsync(float N, float *inA, float *inB, float *outC, cudaStream_t stream){
           float *g_inA, *g_inB, *g_outC; //device memory 
           cudaStream_t stream_0 = stream;
           int size = N*N*sizeof(float);

           cudaMalloc((void**)&g_inA, size); //malloc on device (writes device memory address) 
           cudaMalloc((void**)&g_inB, size);
           cudaMalloc((void**)&g_outC, size);
           
           //Asynchronously copies memory from Host memory..
           cudaMemcpyAsync(g_inA, inA, size, cudaMemcpyHostToDevice, stream_0);
           cudaMemcpyAsync(g_inB, inB, size, cudaMemcpyHostToDevice, stream_0);
           
           //allocate 256 threads per block 
           //Uses multiple blocks to maximize multi-threading performance
           //these values can be differentiated by gpu architecture
           dim3 threadsPerBlock(16,16);
           dim3 numBlocks(N/16,N/16);

           //Start kernel
           matMul<<<numBlocks ,threadsPerBlock ,0 ,stream_0>>>(g_inA, g_inB, outC, N);
           cudaMemcpyAsync(outC, g_outC, size, cudaMemcpyDeviceToHosT, stream_0);
           cudaStreamSynchronize(stream_0);//wait until processes on streak_0 is finished

           //Free malloced resources on gpu
           cudaFree(g_inA);
           cudaFree(g_inB);
           cudaFree(g_outC);

    }

}
