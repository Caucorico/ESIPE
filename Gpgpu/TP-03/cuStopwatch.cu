#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class cuStopwatch{
	// todo: add your internal data structure, all in private
	private:
		cudaEvent_t start_event;
		cudaEvent_t end_event;
		bool is_watching;

	public:
		cuStopwatch();
		~cuStopwatch();
		void start();
		float stop();
};

cuStopwatch::cuStopwatch(){
	cudaEventCreate(&start_event);
	cudaEventCreate(&end_event);
}

cuStopwatch::~cuStopwatch(){
	cudaEventDestroy(start_event);
	cudaEventDestroy(end_event);
}

void cuStopwatch::start(){
	cudaEventRecord(start_event);
}

float cuStopwatch::stop(){
	float elapsed_time;
	cudaEventRecord(end_event);
	cudaEventSynchronize(end_event);
	cudaEventElapsedTime(&elapsed_time, start_event, end_event);
	return elapsed_time;
}
