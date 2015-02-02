#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N,float *residual) {
	// TODO: kernel computation
	//...
	
	extern __shared__ float res_vector[];	

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int index = row*N+col;	
	int res_index = threadIdx.x*blockDim.x+threadIdx.y;
	
	res_vector[res_index] = 0.0;	
	if( row > 0 && row < (N-1) && col > 0 && col < (N-1) ){
		g[index] = 0.25f *(   h[row*N+(col-1)]
								+ h[row*N+(col+1)] 
				 				+ h[(row-1)*N+col] 
				 				+ h[(row+1)*N+col] );
	
		float diff  = g[index]-h[index];
		res_vector[res_index]  = diff*diff;
		
		
		__syncthreads();
	
	}
	for(unsigned int s = blockDim.x*blockDim.x/2; s> 0 ;s>>=1){
		if( res_index < s)
			res_vector[res_index] += res_vector[res_index+s];
		__syncthreads();
	}
	if( res_index == 0){
		 residual[blockIdx.x*gridDim.x+blockIdx.y] = res_vector[0];
	}
}
