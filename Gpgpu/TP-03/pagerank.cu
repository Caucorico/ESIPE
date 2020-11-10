#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "cuStopwatch.cu"

#define COUNT (1<<23)
#define LINK_PER_PAGE 4
#define ERMIX 0.25f
#define MAXINT (4294967295.0f)
#define DAMPING 0.9f
#define EPSILON 0.00000001f
#define MAXPRCOUNT 16
#define INITPROJ 1024

/* ------------ Pagerank computation, GPU part ------------ */

__global__ void pr_init_gpu(float* pr){
    // TODO: fill in initial value for pagerank
}

__global__ void pr_damping_gpu(float* pr){
    // TODO: fill in (1 - damping constant) for pagerank
}

__global__ void pr_iter_gpu(const uint2* links, const float* oldp, float* newp){
    // TODO: add contributions for each link for pagerank
}

__global__ void pr_conv_check_gpu(const float* oldp, const float* newp, uint32_t* conv){
    // TODO: check for convergence against 
}

float pr_compute_gpu(const uint2* links, float* pr){
    // TODO: control GPU computation, returns computation time (in seconds, not counting memory transfer time)
}

/* ------------ Pagerank computation, CPU part ------------ */

__global__ void pr_init_cpu(float* pr){
    // TODO: equivalence of pr_init_gpu on host
}

__global__ void pr_damping_cpu(float* pr){
    // TODO: equivalence of pr_damping_gpu on host
}

void pr_iter_cpu(const uint2* links, const float* oldp, float* newp){
    // TODO: equivalenc of pr_iter_gpu on host
}

void pr_conv_check_cpu(const float* oldp, const float* newp, uint32_t* conv){
    // TODO: equivalence of pr_conv_check_gpu on host
}

float pr_compute_cpu(const uint2* links, float* pr){
    // TODO: equivalence of pr_compute_gpu on host
}

/* ------------ Random graph generation ------------ */

uint32_t randstate;

uint32_t myrand(){
    randstate ^= randstate << 13;
    randstate ^= randstate >> 17;
    randstate ^= randstate << 5;
    return randstate;
}

void seed(){
    randstate = time(NULL);
    for(int i = 0; i < 16; i++) myrand();
    return;
}

void randgen(uint2* links){
    uint32_t state = time(NULL);
    uint32_t *weight = (uint32_t*)malloc(sizeof(uint32_t) * COUNT);
    memset((void*)weight, 0, sizeof(uint32_t) * COUNT);
    uint32_t totalweight = 0;
    uint32_t lcnt = 0;
    
    // Initial five
    for(int i = 0; i < INITPROJ; i++){
        weight[i] = 1;
        for(int j = 0; j < 4; j++){
            links[lcnt].x = i;
            links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
            lcnt++;
        }
    }
    totalweight = INITPROJ;
    
    // Barabasi-Albert with Erdos-Renyi mix-in
    for(uint32_t i = INITPROJ; i < COUNT; i++){
        for(int k = 0; k < LINK_PER_PAGE; k++){
             if(myrand()/MAXINT < ERMIX){
                links[lcnt].x = i;
                links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
                lcnt++;
            }else{
                uint32_t randweight = (uint32_t)(myrand()/MAXINT*totalweight);
                uint32_t idx = 0;
                while(randweight > weight[idx]){
                    randweight -= weight[idx];
                    idx++;
                }
                links[lcnt].x = i;
                links[lcnt].y = idx;
                lcnt++;
                weight[idx]++;
                totalweight++;
            }
        }
    }
    return;
}

/* ------------ Main control ------------ */

void pr_extract_max(const float* pr, float* prmax, uint32_t* prmaxidx){
    for(int i = 0; i < MAXPRCOUNT; i++) prmax[i] = -1.0f;
    for(uint32_t i = 0; i < COUNT; i++){
        if(pr[i] > prmax[MAXPRCOUNT-1]){
            int ptr = 0;
            while(pr[i] <= prmax[ptr]) ptr++;
            float oldval, newval;
            uint32_t oldidx, newidx;
            newval = pr[i];
            newidx = i;
            for(int j = ptr; j < MAXPRCOUNT; j++){
                oldval = prmax[j];
                oldidx = prmaxidx[j];
                prmax[j] = newval;
                prmaxidx[j] = newidx;
                newval = oldval;
                newidx = oldidx;
            }
        }
    }
    return;
}

int main(){
    // Generating random network
    uint2* randlinks;
    cudaHostAlloc((void**)&randlinks, sizeof(uint2)*COUNT*LINK_PER_PAGE, cudaHostAllocDefault);
    seed();
    randgen(randlinks);
    printf("Finished generating graph\n\n");
    
    // Declaration of needed variables and arrays
    float prmax[MAXPRCOUNT];
    uint32_t prmaxidx[MAXPRCOUNT];
    float elapsed;
    float *pagerank;
    float check;
    cudaHostAlloc((void**)&pagerank, sizeof(float)*COUNT, cudaHostAllocDefault);
    
    // Processing by GPU
    elapsed = pr_compute_gpu(randlinks, pagerank);
    printf("GPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    printf("\n");
    
    // Processing by CPU
    elapsed = pr_compute_cpu(randlinks, pagerank);
    printf("CPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    
    // Free memory
    cudaFreeHost(randlinks);
    cudaFreeHost(pagerank);
	return 0;
}