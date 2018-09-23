#include<iostream>
#include<math>

class cudaMatrix{

  private:
        //for 2-dimensional Squre Matrix
        __global__
        static void matMul(float* A, float* B, float*C, int N){

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
        //for Any 2-dimensional matrices
        __global__
        static void matMul(float *A, float*B, float*C, int M, int MN, int N){
            //current row and column for this thread to execute
            int row = blockIdx.y*blockDim.y + threadIdx.y; 
            int col = blockIdx.x*blockDim.x + threadIdx.x;
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
       static void MatMul(float N, float *inA, float *inB, float *outC){

           float *g_inA, *g_inB, *g_outC;
           cudaStream_t stream_0;
           int size = N*N*sizeof(float);

           cudaMalloc((void**)&g_inA, size);
           cudaMalloc((void**)&g_inB, size);
           cudaMalloc((void**)&g_outC, size);
           
           cudaMemcpyAsync(g_inA, inA, size, cudaMemcpyHostToDevice, stream_0);
           cudaMemcpyAsync(g_inB, inB, size, cudaMemcpyHostToDevice, stream_0);
           //cudaMemcpy(g_outC, outC, N*N*sizeof(float));

           dim3 threadsPerBlock(16,16);
           dim3 numBlocks(N/16,N/16);

           matmul<<<numBlocks ,threadsPerBlock ,0 ,stream_0>>>(g_inA, g_inB, outC, N);
           cudaMemcpyAsync(outC, g_outC, N*N*sizeof(float), cudaMemcpyDeviceToHosT, stream_0);
           cudaStreamSynchronize(stream_0);
           cudaFree(g_inA);
           cudaFree(g_inB);
           cudaFree(g_outC);
       }

}
