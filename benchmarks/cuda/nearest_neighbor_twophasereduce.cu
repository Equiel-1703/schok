/*
 * nn.cu
 * Nearest Neighbor
 * Modified by André Du Bois: changed depracated api, creating data set in memory. clean up code not used
 */

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#include <time.h>


__global__
void reduce_kernel(float *ra, float *rout, float (*f)(float,float),
                    float identity, int n)
{
    
    __shared__ float cache[256];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    float temp = identity;
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


__global__
void map_step_2para_1resp_kernel(float *d_array, float *d_result, int step, float par1, float par2, int size, float (*f)(float*,float,float))
{
int globalId = ((blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x)) + threadIdx.x);
int id = (step * globalId);
if((globalId < size))
{
        d_result[globalId] = f((d_array + id), par1, par2);
}

}

__device__
float euclid(float *d_locations, float lat, float lng)
{
return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}

__device__ void* euclid_ptr = (void*) euclid;

extern "C" void* get_euclid_ptr()
{
        void* host_function_ptr;
        cudaMemcpyFromSymbol(&host_function_ptr, euclid_ptr, sizeof(void*));
        return host_function_ptr;
}


__device__
float menor(float x, float y)
{
    if(x<y)
    {
    return (x);
    }
    else{
    return (y);
    }


}

__device__ void* menor_ptr = (void*) menor;

extern "C" void* get_menor_ptr()
{
        void* host_function_ptr;
        cudaMemcpyFromSymbol(&host_function_ptr, menor_ptr, sizeof(void*));
        return host_function_ptr;
}



/*
typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;
*/
void loadData(float *locations, int size);
//void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
//void printUsage();
//int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
//                     int *q, int *t, int *p, int *d);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position

__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng)
{
	//int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
	int globalId = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x; // more efficient
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}
**/
/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
//	int    i=0;
	//float lat=0, lng=0;
	
  //  std::vector<Record> records;
	float *locations;

  int numRecords = atoi(argv[1]);
    
   locations = (float *)malloc(sizeof(float) * 2*numRecords);
   // int numRecords = loadData(filename,records,locations);
   loadData(locations,numRecords);

    
	float *distances;
	//Pointers to device memory
	float *d_locations;
	float *d_distances;


	int n_threads_map = 256;
    int n_blocks_map = (numRecords + n_threads_map - 1)/ n_threads_map;

    int threadsPerBlock = 256;
    int numberOfBlocks  = 64;

    

	/**
	
	* Allocate memory on host and device

  */

  float time;
    cudaEvent_t start, stop;   
     cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;

    //int size_dist = numRecords/2;
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(float) * 2 * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

   /**
    * Transfer data from host to device
    */
    cudaMemcpy( d_locations, &locations[0], sizeof(float) * 2* numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel --
    */

   


    float (*f1)(float*,float,float) = (float (*)(float*,float,float)) get_euclid_ptr();
    float (*f2)(float,float) = (float (*)(float,float)) get_menor_ptr();

    map_step_2para_1resp_kernel<<< n_blocks_map, n_threads_map >>>(d_locations,d_distances,2,0.0,0.0, numRecords, f1);
    

    cudaDeviceSynchronize();

    
    float *resp, *d_resp, *d_partials;
    resp = (float *)malloc(sizeof(float));

	cudaMalloc((void **) &d_resp,sizeof(float));
    //j_error = cudaGetLastError();
    //if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    

    cudaMalloc((void**)&d_partials, numberOfBlocks * sizeof(float));
    //j_error = cudaGetLastError();
    //if (j_error != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(j_error)); exit(1); }
    


    reduce_kernel<<<numberOfBlocks, threadsPerBlock>>>(d_distances, d_partials, f2, 5000.0f, numRecords);
    reduce_kernel<<<1, threadsPerBlock>>>(d_partials, d_resp, f2, 5000.0f, numberOfBlocks);

    cudaDeviceSynchronize();
    //Copy data from device memory to host memory

    cudaMemcpy( resp, d_resp, sizeof(float), cudaMemcpyDeviceToHost );

   // printf("Resposta: %f\n", *resp);
	// find the resultsCount least distances
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);
   
     cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

     printf("CUDA\t%d\t%3.1f\n", numRecords,time);

}

void loadData(float* locations, int size){
   
	for (int i=0;i<size;i++){
			
            locations[0] = ((float)(7 + rand() % 63)) + ((float) rand() / (float) 0x7fffffff);

            locations[1] = ((float)(rand() % 358)) + ((float) rand() / (float) 0x7fffffff); 

            locations = locations +2;
            
           
        }
     
}



