#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>

#include <math.h>
#include <dlfcn.h>
#include <string.h>

#include <chrono>

#define _bitsperpixel 32
#define _planes 1
#define _compression 0
#define _xpixelpermeter 0x13B // 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x13B // 0x130B //2835 , 72 DPI

#pragma pack(push, 1)
typedef struct
{
    uint8_t signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct
{
    uint32_t dibheadersize;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t ypixelpermeter;
    uint32_t xpixelpermeter;
    uint32_t numcolorspallette;
    uint32_t mostimpcolor;
} bitmapinfoheader;
typedef struct
{
    fileheader fileheader;
    bitmapinfoheader bitmapinfoheader;
} bitmap;
#pragma pack(pop)

#define MX_ROWS(matrix) (((uint32_t *)matrix)[0])
#define MX_COLS(matrix) (((uint32_t *)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t *)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t *)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t *)matrix)[0]) * (((uint32_t *)matrix)[1]) + 2)

void genBpm(int h, int w, int *pb)
{
    uint32_t height = (uint32_t)h;
    uint32_t width = (uint32_t)w;

    char *file_name = (char *)"julia_cuda.bmp";
    uint32_t pixelbytesize = height * width * _bitsperpixel / 8;
    uint32_t _filesize = pixelbytesize + sizeof(bitmap);
    FILE *fp = fopen(file_name, "wb");
    bitmap *pbitmap = (bitmap *)calloc(1, sizeof(bitmap));

    int size_pb = h * w * 4;
    uint8_t *pixelbuffer = (uint8_t *)malloc(size_pb);

    for (int i = 0; i < size_pb; i++)
    {
        pixelbuffer[i] = (uint8_t)pb[i];
    }

    // strcpy(pbitmap->fileheader.signature,"BM");
    pbitmap->fileheader.signature[0] = 'B';
    pbitmap->fileheader.signature[1] = 'M';
    pbitmap->fileheader.filesize = _filesize;
    pbitmap->fileheader.fileoffset_to_pixelarray = sizeof(bitmap);
    pbitmap->bitmapinfoheader.dibheadersize = sizeof(bitmapinfoheader);
    pbitmap->bitmapinfoheader.width = width;
    pbitmap->bitmapinfoheader.height = height;
    pbitmap->bitmapinfoheader.planes = _planes;
    pbitmap->bitmapinfoheader.bitsperpixel = _bitsperpixel;
    pbitmap->bitmapinfoheader.compression = _compression;
    pbitmap->bitmapinfoheader.imagesize = pixelbytesize;
    pbitmap->bitmapinfoheader.ypixelpermeter = _ypixelpermeter;
    pbitmap->bitmapinfoheader.xpixelpermeter = _xpixelpermeter;
    pbitmap->bitmapinfoheader.numcolorspallette = 0;
    fwrite(pbitmap, 1, sizeof(bitmap), fp);
    fwrite(pixelbuffer, 1, pixelbytesize, fp);
    fclose(fp);
    free(pbitmap);
}

__device__ int julia(int x, int y, int dim)
{
    float scale = 0.1;
    float jx = ((scale * (dim - x)) / dim);
    float jy = ((scale * (dim - y)) / dim);
    float cr = (-0.8);
    float ci = 0.156;
    float ar = jx;
    float ai = jy;
    for (int i = 0; i < 200; i++)
    {
        float nar = (((ar * ar) - (ai * ai)) + cr);
        float nai = (((ai * ar) + (ar * ai)) + ci);
        if ((((nar * nar) + (nai * nai)) > 1.0e3))
        {
            return (0);
        }

        ar = nar;
        ai = nai;
    }

    return (1);
}

__device__ void *julia_ptr = (void *)julia;

extern "C" void *get_julia_ptr()
{
    void *host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, julia_ptr, sizeof(void *));
    return host_function_ptr;
}

__device__ void julia_function(int *ptr, int x, int y, int dim)
{
    int offset = (x + (y * dim));
    int juliaValue = julia(x, y, dim);
    ptr[((offset * 4) + 0)] = (255 * juliaValue);
    ptr[((offset * 4) + 1)] = 0;
    ptr[((offset * 4) + 2)] = 0;
    ptr[((offset * 4) + 3)] = 255;
}

__device__ void *julia_function_ptr = (void *)julia_function;

extern "C" void *get_julia_function_ptr()
{
    void *host_function_ptr;
    cudaMemcpyFromSymbol(&host_function_ptr, julia_function_ptr, sizeof(void *));
    return host_function_ptr;
}

__global__ void mapgen2D_xy_1para_noret_ker(int *resp, int arg1, int size, void (*f)(int *, int, int, int))
{
    int x = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int y = ((blockIdx.y * blockDim.y) + threadIdx.y);
    if (((x < size) && (y < size)))
    {
        f(resp, x, y, arg1);
    }
}

void print_gpu_info()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return;
    }

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version:  %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("Total Devices:        %d\n\n", deviceCount);

    // 2. Iterate through each GPU and get properties
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        cudaSetDevice(dev);

        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);

        printf("================ Device %d: %s ================\n", dev, prop.name);
        printf("Compute Capability:       %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory:      %.2f GB (%zu bytes)\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024), prop.totalGlobalMem);
        printf("Free Global Memory:       %.2f GB (%zu bytes)\n", (double)freeMem / (1024 * 1024 * 1024), freeMem);
        printf("Multiprocessors (SMs):    %d\n", prop.multiProcessorCount);
        printf("Max Threads per Block:    %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per SM:       %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Warp Size:                %d\n", prop.warpSize);
        printf("Shared Memory per Block:  %.2f KB\n", (double)prop.sharedMemPerBlock / 1024.0);
        printf("Memory Bus Width:         %d-bit\n", prop.memoryBusWidth);
        printf("L2 Cache Size:            %.2f MB\n\n", (double)prop.l2CacheSize / (1024.0 * 1024.0));
    }

    // Set device 0 as the default device for subsequent CUDA operations
    cudaSetDevice(0);
}

int main(int argc, char const *argv[])
{
    print_gpu_info();

    size_t usr_value = (size_t)atol(argv[1]);

    size_t height, width, DIM;
    height = width = DIM = usr_value;

    size_t size_array = sizeof(int) * height * width * 4;

    cudaError_t j_error;

    // int pixelbytesize=  height*width*_bitsperpixel/8;
    printf("IMG size = %d x %d\n", (int)height, (int)width);
    printf("IMG size in bytes = %lu\n", size_array);

    int *d_pixelbuffer;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    ////////
    auto start_alloc = std::chrono::steady_clock::now();
    cudaMalloc((void **)&d_pixelbuffer, size_array);
    auto end_alloc = std::chrono::steady_clock::now();

    double alloc_ms = std::chrono::duration<double, std::milli>(end_alloc - start_alloc).count();

    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 1: %s\n", cudaGetErrorString(j_error));

    printf("Time taken for cudaMalloc: %f ms\n", alloc_ms);
    ////////

    ////////////////////
    const int blockSize = 16;
    const int gridSize = (DIM + blockSize - 1) / blockSize;

    dim3 block(blockSize, blockSize);
    dim3 grid(gridSize, gridSize);

    void (*f)(int *, int, int, int) = (void (*)(int *, int, int, int))get_julia_function_ptr();

    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", grid.x, grid.y, block.x, block.y);

    auto start_kernel = std::chrono::steady_clock::now();
    mapgen2D_xy_1para_noret_ker<<<grid, block>>>(d_pixelbuffer, DIM, DIM, f);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::steady_clock::now();

    double kernel_ms = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    printf("Time taken for kernel execution: %f ms\n", kernel_ms);

    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 3: %s\n", cudaGetErrorString(j_error));
    ////////

    int *h_pixelbuffer = (int *)malloc(size_array);

    auto start_memcpy = std::chrono::steady_clock::now();
    cudaMemcpy(h_pixelbuffer, d_pixelbuffer, size_array, cudaMemcpyDeviceToHost); // return results
    auto end_memcpy = std::chrono::steady_clock::now();

    double copy_ms = std::chrono::duration<double, std::milli>(end_memcpy - start_memcpy).count();

    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 7: %s\n", cudaGetErrorString(j_error));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Time taken for cudaMemcpy: %f ms\n", copy_ms);
    printf("CUDA\t%lu\t%3.1f\n", usr_value, time);
    printf("Total time (chrono): %f ms\n", alloc_ms + kernel_ms + copy_ms);

    genBpm(height, width, h_pixelbuffer);

    free(h_pixelbuffer);
    cudaFree(d_pixelbuffer);
}
