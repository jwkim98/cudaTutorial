#ifndef __GPU_LIB__
#define __GPU_LIB__

#define PAGE_SIZE 1024*4
#include <stdlib.h>
#include <functional>

namespace cuda_lib{
    class cudaMatrix{

      private:
            //for 2-dimensional Squre Matrix
            void MatMul(float* A, float* B, float* C, int N);

            template<typename data_type>
            void MatMulUnified(int N ,data_type* A ,data_type* B, data_type* C);

            template <typename data_type>
            void MatMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C);

            //for Any 2-dimensional matrices
            template <typename data_type>
            void MatOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate);

            template <typename data_type>
            void MatAdd(int N, data_type *A, data_type *B, data_type *C);
            
            template <typename data_type>
            void MatDot(int N, data_type* A, data_type* B, data_type* C);

            template <typename data_type>
            void MatTranspose(int M, int N, data_type* A, data_type* B);

      public:
            void MatMulAsync(float N, float *inA, float *inB, float *outC, cudaStream_t stream); 
    };


    __global__ void matMul(float* A, float* B, float* C, int N);

    template<typename data_type>
    __global__ void matMulUnified(int N ,data_type* A ,data_type* B, data_type* C);

    template <typename data_type>
    __global__ void matMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C);

    //for Any 2-dimensional matrices
    template <typename data_type>
    __global__ void matOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate);

    template <typename data_type>
    __global__ void matAdd(int N, data_type *A, data_type *B, data_type *C);
    
    template <typename data_type>
    __global__ void matDot(int N, data_type* A, data_type* B, data_type* C);

    template <typename data_type>
    __global__ void matTranspose(int M, int N, data_type* A, data_type* B);



    void cudaMatrix::MatMul(float* A, float* B, float* C, int N){
        matMul<<<1,1>>>(A,B,C,N);
    }

    template<typename data_type>
    void cudaMatrix::MatMulUnified(int N, data_type* A, data_type* B, data_type* C){
        matMulUnified(N, A, B, C);
    }

    template <typename data_type>
    void MatMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C){
        matMulTwo(M, MN, N, input_A, input_B, output_C);
    }


    //for Any 2-dimensional matrices
    template <typename data_type>
    void cudaMatrix::MatOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate){
        matop(M, N, input, output, accumulate);
    }

    template <typename data_type>
    void cudaMatrix::MatAdd(int N, data_type *A, data_type *B, data_type *C){
        matAdd(N, A, B, C);
    }
    
    template <typename data_type>
    void cudaMatrix::MatDot(int N, data_type* A, data_type* B, data_type* C){
        matDot(N, A, B, C);
    }

    template <typename data_type>
    void cudaMatrix::MatTranspose(int M, int N, data_type* A, data_type* B){
        matTranspose(M, N, A, B);
    }
}
#endif
