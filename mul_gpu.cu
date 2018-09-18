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

           cudaMalloc((void**)&g_inA, N*N*sizeof(float), cudaMemcpyHostToDevice);
           cudaMalloc((void**)&g_inB, N*N*sizeof(float), cudaMemcpyHostToDevice);
           cudaMalloc((void**)&g_outC, N*N*sizeof(float), cudaMemcpyHostToDevice);
           
           cudaMemcpy(g_inA, inA, N*N*sizeof(float));
           cudaMemcpy(g_inB, inB, N*N*sizeof(float));
           //cudaMemcpy(g_outC, outC, N*N*sizeof(float));

           matmul<<<1,256>>>(inA, inB, outC, N);

           cudaMemcpy(outC, g_outC, N*N*sizeof(float), cudaMemcpyDeviceToHost);

       }



}
