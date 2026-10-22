#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Double-precision version of dotprod_twopass.cu.
 * Same structure: map kernel, then reduce_kernel launched twice
 * (N elements -> numberOfBlocks partials -> 1 value).
 * The identity element is a parameter so the reduction stays generic in f.
 */

// ---------------------------------------------------------------------
// map: a3[i] = f(a1[i], a2[i])
// ---------------------------------------------------------------------
__global__
void map_2kernel(double *a1, double *a2, double *a3, int size,
                  double (*f)(double,double))
{
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size) {
      a3[id] = f(a1[id],a2[id]);
    }
}

// ---------------------------------------------------------------------
// reduce: one block-level reduction; each block writes out[blockIdx.x].
// Launched twice from the host (see main).
// ---------------------------------------------------------------------
__global__
void reduce_kernel(double *ra, double *rout, double (*f)(double,double),
                    double identity, int n)
{
    
    __shared__ double cache[256];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    double temp = identity;
    while (tid < n) {
        temp = f(ra[tid], temp);
        tid = tid + (blockDim.x * gridDim.x);
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i)
            cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex]);
        __syncthreads();
        i = i / 2;
    }

    if (cacheIndex == 0)
        rout[blockIdx.x] = cache[0];
}

//#############################

__device__
double anonymous_mult(double a, double b) { return a * b; }

__device__ void* anonymous_mult_ptr = (void*) anonymous_mult;

extern "C" void* get_anonymous_mult_ptr()
{
    void* host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, anonymous_mult_ptr, sizeof(void*));
    return host_function_ptr;
}

//#############################

__device__
double anonymous_sum(double a, double b) { return a + b; }

__device__ void* anonymous_sum_ptr = (void*) anonymous_sum;

extern "C" void* get_anonymous_sum_ptr()
{
    void* host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, anonymous_sum_ptr, sizeof(void*));
    return host_function_ptr;
}

//#############################

int main(int argc, char *argv[])
{
    double *a, *b, *final;
    double *dev_a, *dev_b, *dev_resp, *d_partials, *d_final;
    cudaError_t j_error;

    int N = atoi(argv[1]);

    a = (double*)malloc((size_t)N * sizeof(double));
    b = (double*)malloc((size_t)N * sizeof(double));

    for (int i = 0; i < N; i++) a[i] = 1; //rand();
    for (int i = 0; i < N; i++) b[i] = 1; //rand();

    int n_threads_map = 256;
    int n_blocks_map = (N + n_threads_map - 1)/ n_threads_map;


    int threadsPerBlock = 256;
    int numberOfBlocks  = 64;  // fixed, occupancy-sized grid; not tied to N



    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&dev_a, (size_t)N * sizeof(double));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&dev_b, (size_t)N * sizeof(double));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&dev_resp, (size_t)N * sizeof(double));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&d_partials, numberOfBlocks * sizeof(double));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMalloc((void**)&d_final, sizeof(double));
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    cudaMemcpy(dev_a, a, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    cudaMemcpy(dev_b, b, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    double (*f1)(double,double) = (double (*)(double,double)) get_anonymous_mult_ptr();
    double (*f2)(double,double) = (double (*)(double,double)) get_anonymous_sum_ptr();

    map_2kernel<<<n_blocks_map, n_threads_map>>>(dev_a, dev_b, dev_resp, N, f1);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    // pass 1: N elements -> numberOfBlocks partials
    // identity for `+` is 0.0; use -INFINITY for max, +INFINITY for min, etc.
    reduce_kernel<<<numberOfBlocks, threadsPerBlock>>>(dev_resp, d_partials, f2, 0.0, N);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    // pass 2: numberOfBlocks partials -> 1 final value (single block)
    reduce_kernel<<<1, threadsPerBlock>>>(d_partials, d_final, f2, 0.0, numberOfBlocks);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
     
    final = (double*)malloc(sizeof(double));
    cudaMemcpy(final, d_final, sizeof(double), cudaMemcpyDeviceToHost);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resp);
    cudaFree(d_partials);
    cudaFree(d_final);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Result: %f\n", *final);
    printf("CUDA-double\t%d\t%3.1f\n", N, time);

    free(a);
    free(b);
    free(final);

    return 0;
}
