#include <iostream>
#include <math>
#include <stdlib.h>
#define PAGE_SIZE = 1024*4

class cudaMatrix{

  private:
        //for 2-dimensional Squre Matrix
        __global__
        static void matMul(float* A, float* B, float* C, int N){

            int row = blockIdx.y*blockDim.y + threadIdx.y;
            int col = blockIdx.x*blockDim.x + threadIdx.x;
            if (row < N && col < N){
                float sum = 0.0f;
                for(int i=0; i < N; i++){
                    sum += A[row*N + i] * B[col + i*N];
                }
                C[row*N + col] = sum;
            }
        }

        template <typename data_type>
        __global__
        static void matMulUnified(uint32_t rows ,data_type* A ,data_type* B, data_type* C){
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            size_t tidRow = tid/row;
            size_t tidCol = tid%row; 
            size_t n = rows;
            size_t data_size = sizeof(data_type);

            if(tidRow < rows && tidCol < rows){
                data_type accum = 0;
                for(tid; tid < n; tid += blockDim.x * gridDim.x){
                    for(int i = 0; i < rows; i++){
                        uint32_t indexRow = tid*tidRow + i;
                        uint32_t indexCol = i*rows + tidCol;
                        accum += A[indexRow] * B[indexCol];
                    }

                    data_type*C[tidRow*row + tidCol] = accum;
                }
            }
        }

        //for Any 2-dimensional matrices
        __global__
        static void matMul(float *A, float*B, float*C, int M, int MN, int N){
            //current row and column for this thread to execute
            int row = blockIdx.y*blockDim.y + threadIdx.y; 
            int col = blockIdx.x*blockDim.x + threadIdx.x;
            int lane_id = (blockDim.x*blockIdx.x + blockIdx.y)>>5; //devide by 32
            int warps_per_grid = (blockDim.x*gridDim.x*blockDim.y*gridDim*y) >>5;

            //TODO devide the memory access in pages

            //Result would be M*N matrix
            if(row < M && col < N){
                float sum = 0.0f;
                for(int i=0; i < MN; i++){
                    sum += A[row*MN + i] * B[col + i*N];
                }
                C[row*N + col] = sum;
            }            
        }

        __global__
        static void matAdd(int N, float *A, float *B, float *C){
        
            int index = threadIdx.x;
            int stride = blockDim.x;
            for(int i=index; i < N ; i += stride)
                C[i] = A[i] + B[i];
        }

       __global__
       static void matDot(int N, float* A, float* B, float* C){
       
            int index = threadIdx.x;
            int stride = blockDim.x;
            for(int i=index; i < n ; i += stride)
                C[i] = A[i] * B[i];
       }

       __global__
       static void matTranspose(int M, int N, float* A, float*B){
           //current row and column for this thread to execute
           int row = blockIdx.y*blockDim.y + threadIdx.y;
           int col = blockIdx.x*blockDim.x + threadIdx.x;
           if(row<M && col<N)
               B[col*M + row] = A[row*N + col];
       }

  public:
       static void MatMulAsync(float N, float *inA, float *inB, float *outC, cudaStream_t){

           float *g_inA, *g_inB, *g_outC; //device memory 
           cudaStream_t stream_0;
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
           matmul<<<numBlocks ,threadsPerBlock ,0 ,stream_0>>>(g_inA, g_inB, outC, N);
           cudaMemcpyAsync(outC, g_outC, size, cudaMemcpyDeviceToHosT, stream_0);
           cudaStreamSynchronize(stream_0);//wait until processes on streak_0 is finished

           //Free malloced resources on gpu
           cudaFree(g_inA);
           cudaFree(g_inB);
           cudaFree(g_outC);
       }

       static void MatMulAsyncUnified(float N, float *inA, float *inB, float *outC){
          
       }

}
