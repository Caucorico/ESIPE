#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class cuStopwatch{
    // todo: add your internal data structure, all in private
    
    public:
        cuStopwatch();
        ~cuStopwatch();
        int start();
        float stop();
};

cuStopwatch::cuStopwatch(){
    // todo: constructor
}

cuStopwatch::~cuStopwatch(){
    // todo: destructor
}

void cuStopwatch::start(){
    // todo: start the stopwatch, and ignore double start
}

float cuStopwatch::stop(){
    // todo: stop the stopwatch and return elapsed time, ignore invalid stops (e.g. stop when not yet started or double stop)
}