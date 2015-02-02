#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	// TODO: kernel computation
	//...
	/*int tidx = blockIdx.x*blockDim.x+threadIdx.x;
	int tidy = blockIdx.y*blockDim.y+threadIdx.y;

	if( tidx > 0 && tidx < (N-1) && tidy > 0 && tidy < (N-1) )
		g[tidx*N+tidy] = 0.25*(   h[tidx*N+(tidy-1)]
					+ h[tidx*N+(tidy+1)] 
				 	+ h[(tidx-1)*N+tidy] 
				 	+ h[(tidx+1)*N+tidy] );
	*/
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if( row > 0 && row < (N-1) && col > 0 && col < (N-1) )
		g[row*N+col] = 0.25f *(   h[row*N+(col-1)]
								+ h[row*N+(col+1)] 
				 				+ h[(row-1)*N+col] 
				 				+ h[(row+1)*N+col] );
		
}
