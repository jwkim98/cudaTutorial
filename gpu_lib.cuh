#ifndef __GPU_LIB__
#define __GPU_LIB__

#define PAGE_SIZE 1024*4
#include <cstdlib>
#include <functional>
#include <iostream>

namespace cuda_lib{
    class cudaMatrix{

      private:
            //for 2-dimensional Squre Matrix
            void MatMul(int N, float* A, float* B, float* C);

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
            cudaMatrix(){
                std::cout<<"cudaMatrix Initialized"<<std::endl;
            }

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



    void cudaMatrix::MatMul(int N, float* A, float* B, float* C){
        dim3 block(32,32);
        dim3 grid(N/32+1, N/32+1);
        matMul<<<grid,block>>>(A,B,C,N);
    }

    template<typename data_type>
    void cudaMatrix::MatMulUnified(int N, data_type* A, data_type* B, data_type* C){
        dim3 block(32,32);
        dim3 grid(N/32+1, N/32+1);
        matMulUnified<<<grid,block>>>(N, A, B, C);
    }

    template <typename data_type>
    void MatMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C){
        dim3 block(32,32);
        dim3 grid(M/32+1, N/32+1);
        matMulTwo<<<grid,block>>>(M, MN, N, input_A, input_B, output_C);
    }


    //for Any 2-dimensional matrices
    template <typename data_type>
    void cudaMatrix::MatOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate){
        dim3 block(32,32);
        dim3 grid(M/32+1, N/32+1);
        matop<<<grid,block>>>(M, N, input, output, accumulate);
    }

    template <typename data_type>
    void cudaMatrix::MatAdd(int N, data_type *A, data_type *B, data_type *C){
        dim3* block = new dim3(32,32);
        dim3 grid(N/32+1, N/32+1);
        matAdd<<<grid,*block>>>(N, A, B, C);
        delete block;
    }
    
    template <typename data_type>
    void cudaMatrix::MatDot(int N, data_type* A, data_type* B, data_type* C){
        dim3 block(32,32);
        dim3 grid(N/32+1, N/32+1);
        matDot<<<grid,block>>>(N, A, B, C);
    }

    template <typename data_type>
    void cudaMatrix::MatTranspose(int M, int N, data_type* A, data_type* B){
        dim3 block(32,32);
        dim3 grid(M/32+1, N/32+1);
        matTranspose<<<grid,block>>>(M, N, A, B);
    }
}
#endif
