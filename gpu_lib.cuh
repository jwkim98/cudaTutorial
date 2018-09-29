/*
   gpu_lib.cuh
*/

#ifndef __GPU_LIB__
#define __GPU_LIB__

#define PAGE_SIZE 1024*4 //default page size for intel x86 systems
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>

using namespace std;

namespace cuda_lib{
    class cudaException{
        public:
            char* message;
            cudaException(char *msg){
                message = new char[strlen(msg)+1];
                strcpy(message, message);
            }
            cudaException(const cudaException &ref){ //copy constructor
                this->message = new char[strlen(ref.message)+1];
                strcpy(this->message, ref.message);
            }
            ~cudaException(){
                delete[] message;
            }
    };

    template <typename T>
    class Matrix
    {
      private:
            uint32_t sizeX;
            uint32_t sizeY;
            uint32_t total_size;
      public:
            T *A; //data storing actual matrix
            Matrix(T* A, uint32_t x, uint32_t y = 1): sizeX(x), sizeY(y)
            {
                if(x == 0)
                    throw cudaException("x cannot be 0");                
                if(y != 0)
                    total_size = x*y;
                else
                    total_size = x;
                A = new T[total_size];
                memcpy(this->A, A, total_size); //do deep copy
            }

            Matrix(const Matrix &matrix): sizeX(matrix.sizeX), sizeY(matrix.sizeY), total_size(matrix.total_size)// copy constructor for deep copy
            {
                A = new T[total_size];
                memcpy(A, matrix.A, total_size); // do deep copy
            }

            Matrix& operator=(const Matrix &matrix);

            Matrix operator+(const Matrix &matrix);

            Matrix operator-(const Matrix &matrix);

            Matrix operator*(const Matrix &matrix);

            Matrix operator*(int mul);

            friend Matrix operator*(int mul, const Matrix &matrix);

            Matrix accumulate(function<T(T)> fc);

            Matrix& operator+=(const Matrix &matrix);

            Matrix& operator-=(const Matrix &matrix);

            Matrix dot(const Matrix &matrix);

            Matrix& Transpose();

            ~Matrix()
            {
                delete[] A; //free allocated array of T
            }

            int sizeX(){
                return sizeX;
            }

            int sizeY(){
                return sizeY;
            }

            Matrix& reshape();

    };

        template <typename T>
        Matrix<T>& Matrix<T>::operator=(const Matrix<T> &matrix)
        {
            sizeX = matrix.sizeX;
            sizeY = matrix.sizeY;
            total_size = matrix.total_size;
            A = new T[total_size];
            memcpy(A, matrix.A, total_size); // do deep copy
            return *this;
        }

        template <typename T>
        Matrix<T>& Matrix<T>::operator+(const Matrix<T> &matrix){
            
        }



    class cudaMatrix{

      private:
            //for 2-dimensional Squre Matrix
            template<typename data_type>
            void MatMul(int N, data_type* A, data_type* B, data_type* C);

            template<typename data_type>
            void MatMulUnified(int N ,data_type* A ,data_type* B, data_type* C);

            template <typename data_type>
            void MatMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C);

            //for Any 2-dimensional matrices
            template <typename data_type>
            void MatOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate);

            template <typename data_type>
            void MatAdd(int M, int N, int total_Size, data_type* A, data_type* B, data_type *C, cudaStream_t stream);
            
            template <typename data_type>
            void MatDot(int N, data_type* A, data_type* B, data_type* C);

            template <typename data_type>
            void MatTranspose(int M, int N, data_type* A, data_type* B);

      public:
            cudaMatrix();

            cudaMatrix(int devicenum);

            void MatMulAsync(int N, float *inA, float *inB, float *outC, cudaStream_t stream);
    };

    cudaMatrix::cudaMatrix(){
        cudaDeviceProp* prop = NULL;
        int* device = NULL;
        cudaGetDevice(device);
        cudaGetDeviceProperties(prop, *device);
        cout<< "Device Name:"<< prop->name << endl << "Total memory:"<< prop->totalGlobalMem << endl;
    }

    cudaMatrix::cudaMatrix(int devicenum){
        cudaDeviceProp* prop = NULL;
        cudaGetDeviceProperties(prop, devicenum);
        cout<< "Device Name:"<< prop->name << endl <<"Total memory:" << prop->totalGlobalMem << endl;
    }

    template<typename data_type>
    __global__ void matMul(int N, data_type* A, data_type* B, data_type* C);

    template<typename data_type>
    __global__ void matMulUnified(int N ,data_type* A ,data_type* B, data_type* C);

    template <typename data_type>
    __global__ void matMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C);

    //for Any 2-dimensional matrices
    template <typename data_type>
    __global__ void matOp(int M ,int N ,data_type *input ,data_type* output ,std::function<data_type (data_type)> accumulate);

    template <typename data_type>
    __global__ void matAdd(int M, int N, data_type *A, data_type *B, data_type *C);
    
    template <typename data_type>
    __global__ void matDot(int N, data_type* A, data_type* B, data_type* C);

    template <typename data_type>
    __global__ void matTranspose(int M, int N, data_type* A, data_type* B);


    template<typename data_type>
    void cudaMatrix::MatMul(int N, data_type* A, data_type* B, data_type* C){
        dim3 block(32,32);
        dim3 grid(N/32+1, N/32+1);
        matMul<<<grid,block>>>(N,A,B,C);
    }

    template<typename data_type>
    void cudaMatrix::MatMulUnified(int N, data_type* A, data_type* B, data_type* C){
        dim3 block(32,32);
        dim3 grid(N/32+1, N/32+1);
        matMulUnified<<<grid,block>>>(N, A, B, C);
    }

    template <typename data_type>
    void cudaMatrix::MatMulTwo(int M ,int MN ,int N ,data_type *input_A ,data_type* input_B ,data_type* output_C){
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
    void cudaMatrix::MatAdd(int M, int N, int total_Size, data_type* A, data_type* B, data_type *C, cudaStream_t stream){
        data_type *g_A, *g_B, *g_out;
        int size = sizeof(total_size);
        cudaMalloc((void**)&g_A, size);
        cudaMalloc((void**)&g_B, size);
        cudaMalloc((void**)&g_out, size);

        cudaMemcpyAsync(g_A, A, size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(g_B, B, size, cudaMemcpyHostToDevice, stream);
        dim3 block(32,32);
        dim3 grid(M/32+1, N/32+1);
        matAdd<<<grid,block,0,stream>>>(M, N, g_A, g_B, g_out);
        cudaMemcpyAsync(C, g_out, size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(g_A);
        cudaFree(g_B);
        cudaFree(g_out);

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
