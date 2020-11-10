#include "SDL.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LEN 1024
#define LENSHIFT 10
#define ITERMAX 1024
#define getindex(i, j) (((i)<<LENSHIFT)+(j))
#define NCOLOR 64
#define NCOLORMASK 63

SDL_Window *screen;
SDL_Renderer *ren;
SDL_Texture *tex;
SDL_Surface *mysurf;

uint32_t iterscpu[LEN*LEN];
uint32_t colors[NCOLOR+1];
uint32_t* iters;

void iterate_cpu(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    // todo: write the CPU version of iteration
    return;
}

__global__ void iterate_gpu(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    // todo: write the GPU kernel of iteration
    return;
}

void kernel_call(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    // todo: write the kernel call here, with given parameters and appropriate thread grid configurations
    return;
}

void generate_colors(const SDL_PixelFormat* format){
    double h = 0.0;
    for(int i=0; i<NCOLOR; i++){
        int ph = h / 60;
        float f = (h/60.0 - ph);
        int v = 255;
        int p = 64;
        int q = (int)(255*(1 - f*0.75f));
        int t = (int)(255*(0.25f + f*0.75f));
        switch(ph){
            case 0:
                colors[i] = SDL_MapRGB(format, v, t, p);
                break;
            case 1:
                colors[i] = SDL_MapRGB(format, q, v, p);
                break;
            case 2:
                colors[i] = SDL_MapRGB(format, p, v, t);
                break;
            case 3:
                colors[i] = SDL_MapRGB(format, p, q, v);
                break;
            case 4:
                colors[i] = SDL_MapRGB(format, t, p, v);
                break;
            case 5:
                colors[i] = SDL_MapRGB(format, v, p, q);
                break;
            default:
                break;
        }
        h += 360.0/NCOLOR;
    }
    colors[NCOLOR] = SDL_MapRGB(format, 0, 0, 0);
    return;
}

int main(int argc, char** argv){
    SDL_Event e;
    bool usegpu = false;
    if(argc > 1){
        usegpu = (strcmp(argv[1], "gpu") == 0);
    }
    uint32_t* gpuarray;
    uint32_t* hostarray;
    
    // Initialize SDL
    if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
	atexit(SDL_Quit);
    // Create window
	screen = SDL_CreateWindow("Mandelbrot", 
                        SDL_WINDOWPOS_UNDEFINED,
                        SDL_WINDOWPOS_UNDEFINED,
                        LEN, LEN, SDL_WINDOW_SHOWN);
    if ( screen == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    
    // Initialize CUDA
    if(usegpu){
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaMalloc((void**)&gpuarray, LEN*LEN*sizeof(uint32_t));
        cudaHostAlloc((void**)&hostarray, LEN*LEN*sizeof(uint32_t), cudaHostAllocDefault);
    }
    
    // Create renderer and texture
    SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32);
    generate_colors(fmt);
    ren = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex = SDL_CreateTexture(ren, fmt->format, SDL_TEXTUREACCESS_STREAMING, LEN, LEN);
    
    // Timing
    float totaltime = 0.0f;
    uint32_t frames = 0;
    
    // Window for Mandelbrot
    double targetx = -0.743643887037158704752191506114774;
    double targety = 0.131825904205311970493132056385139;
    double centerx = 0.0;
    double centery = 0.0;
    double delta = 4.0/LEN;
    const double scale = 0.94;
    uint32_t itermax = 32;
    const uint32_t iterstep = 8;
    
    while(true){
        bool flag = false;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT){
                flag = true;
            }
        }
        if(flag) break;
        clock_t t;
        float tsec;
        t = clock();
        // renderer
        if(!usegpu){
            iterate_cpu(iterscpu, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            iters = iterscpu;
        }else{
            kernel_call(gpuarray, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            cudaMemcpyAsync(hostarray, gpuarray, LEN * LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            iters = hostarray;
        }
        
        int len = LEN;
        uint32_t* surf = NULL;
        SDL_LockTexture(tex, NULL, (void**)(&surf), &len);
        for(uint32_t i=0; i<LEN*LEN; i++){
                if (iters[i] < itermax){
                    surf[i] = colors[iters[i]&NCOLORMASK];
                }else{
                    surf[i] = colors[NCOLOR];
                }
        }
        SDL_UnlockTexture(tex);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
        centerx = targetx + (centerx - targetx)*scale;
        centery = targety + (centery - targety)*scale;
        delta *= scale;
        itermax += iterstep;
        t = clock() - t;
        tsec = ((float)t)/CLOCKS_PER_SEC;
        totaltime += tsec;
        tsec = 1.0f/60 - tsec;
        if(tsec > 0) SDL_Delay((uint32_t)(tsec*1000));
        frames++;
        if(frames>=530) break;
    }
    
    char s[100];
    sprintf(s, "Average FPS: %.1f\nFrame count: %u", frames/totaltime, frames);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Benchmark", s, screen);
    SDL_FreeFormat(fmt);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(screen);
    if(usegpu){
        cudaFree(gpuarray);
        cudaFreeHost(hostarray);
    }
    exit(0);
}