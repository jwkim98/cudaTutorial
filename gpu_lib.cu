#include "gpu_lib.cuh"

namespace cuda_lib{

        __global__ void matMul(float* A, float* B, float* C, int N){

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

        template<typename data_type>
        __global__ void matMulUnified(int N ,data_type* A ,data_type* B, data_type* C){
            size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
            size_t tidRow = tid/N;
            size_t tidCol = tid%N; 
            size_t n = N*N;

            if(tidRow < N && tidCol < N){
                data_type accum = 0;
                for(tid; tid < n; tid += blockDim.x * gridDim.x){
                    for(int i = 0; i < N; i++){
                        int indexRow = tid*tidRow + i;
                        int indexCol = i*N + tidCol;
                        accum += A[indexRow] * B[indexCol];
                    }
                    C[tidRow*N + tidCol] = accum;
                }
            }
        }

        template <typename data_type>
        __global__  void matMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C){
            int row = blockIdx.y*blockDim.y + threadIdx.y; //row for this thread to calculate
            int col = blockIdx.x*blockDim.x + threadIdx.x; //column for this thread to calculate
           

            //Result would be M*N matrix
            if(row < M && col < N){
                data_type sum = 0;
                for(int i=0; i < MN; i++){
                    sum += input_A[row*MN + i] * input_B[col + i*N];
                }
                output_C[row*N + col] = sum;
            }
            //TODO think about matrices exceeding the size of whold grid..
        }

        //for Any 2-dimensional matrices
        template <typename data_type>
        __global__ void matOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate){
            //current row and column for this thread to execute
            int row = blockIdx.y*blockDim.y + threadIdx.y; //row for this thread to calculate
            int col = blockIdx.x*blockDim.x + threadIdx.x; //column for this thread to calculate

            int linearId= blockDim.x * gridDim.x * row +col;

            //I made whole 2-dimensional array linear, and mapped it to warp ID by dividing it into 32
            int warp_id = linearId>>5; //devide by 32
            int warps_per_grid = (blockDim.x * gridDim.x * blockDim.y * gridDim.y) >>5; //whole dimension of each grid devided by 32(warp size)
            size_t warps_Required = (M*N * sizeof(data_type) + PAGE_SIZE -1) / PAGE_SIZE;

            int lane_id = linearId%32;

            for(; warp_id < warps_Required ; warp_id += warps_per_grid){
                for(int i=0 ;i < PAGE_SIZE/sizeof(data_type); i += sizeof(data_type)*32){
                    int index = i * warp_id + lane_id;
                    output[index] = accumulate(input[index]); //do some accumulations
                }
            }
        }

        template <typename data_type>
        __global__ void matAdd(int N, data_type *A, data_type *B, data_type *C){
        
            int index = threadIdx.x;
            int stride = blockDim.x;
            for(int i=index; i < N ; i += stride)
                C[i] = A[i] + B[i];
        }

        template <typename data_type>
        __global__ void matDot(int N, data_type* A, data_type* B, data_type* C){
       
            int index = threadIdx.x;
            int stride = blockDim.x;
            for(int i=index; i < N ; i += stride)
                C[i] = A[i] * B[i];
       }

        template <typename data_type>
        __global__ void matTranspose(int M, int N, data_type* A, data_type* B){
           //current row and column for this thread to execute
           int row = blockIdx.y*blockDim.y + threadIdx.y;
           int col = blockIdx.x*blockDim.x + threadIdx.x;
           if(row<M && col<N)
               B[col*M + row] = A[row*N + col];
       }

    

    void cudaMatrix::MatMulAsync(float N, float *inA, float *inB, float *outC, cudaStream_t stream){
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
           cudaMemcpyAsync(outC, g_outC, size, cudaMemcpyDeviceToHost, stream_0);
           cudaStreamSynchronize(stream_0);//wait until processes on streak_0 is finished

           //Free malloced resources on gpu
           cudaFree(g_inA);
           cudaFree(g_inB);
           cudaFree(g_outC);

    }

}

/*
int main(){
    return 0;
}
*/
