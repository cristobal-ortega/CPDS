#include "heat.h"
#include <mpi.h>

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
    int myid, numprocs;
    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);


    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;

    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }
	

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    	{
	    // Computing "Red" blocks
	    for (int ii=0; ii<nbx; ii++) {
		lsw = ii%2;
		for (int jj=lsw; jj<nby; jj=jj+2) 
		    for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
			for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
			    unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
					      u[ i*sizey	+ (j+1) ]+  // right
					      u[ (i-1)*sizey	+ j     ]+  // top
					      u[ (i+1)*sizey	+ j     ]); // bottom
			    diff = unew - u[i*sizey+ j];
			    sum += diff * diff; 
			    u[i*sizey+j]=unew;
			}
	    }
	    
	    // Computing "Black" blocks
	    for (int ii=0; ii<nbx; ii++) {
		lsw = (ii+1)%2;
		for (int jj=lsw; jj<nby; jj=jj+2) 
		    for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
			for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
			    unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
					      u[ i*sizey	+ (j+1) ]+  // right
					      u[ (i-1)*sizey	+ j     ]+  // top
					      u[ (i+1)*sizey	+ j     ]); // bottom
			    diff = unew - u[i*sizey+ j];
			    sum += diff * diff; 
			    u[i*sizey+j]=unew;
			}
	    }
	}

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int myid, numprocs;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
    nbx = NB*numprocs;
    bx = sizex;
    nby = NB*numprocs;
    by = sizey/nby;

	/*DEBUG INFO
		printf("---------------INFO FROM %i---------------\n", myid);
		printf("nbx: %i bx: %i nby: %i by: %i\n", nbx,bx,nby,by);
		printf("sizex: %i, sizey: %i \n", sizex, sizey);
		printf("numprocs: %i, myid: %i \n", numprocs, myid);
	*/

    for (int jj=0; jj<nby; jj++){
		if(myid > 0)
		{
		    //printf("Receiving %i doubles from %i at %i\n", by, myid-1, jj*by);
			// Add +1 in the matrix to avoid sending the borders and easily avoid to treat to send them
			// we are sending in chunks of 16 (by), that is not multiple of sizey (by*nby+2) because of the borders	
			MPI_Recv(&u[1+jj*by], by, MPI_DOUBLE, myid-1, 0, MPI_COMM_WORLD, &status);
		}
    	for (int i=1; i<sizex-1; i++) 
        	for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (  u[ i*sizey	+ (j-1) ]+  // left
				      			u[ i*sizey	+ (j+1) ]+  // right
		 						u[ (i-1)*sizey	+ j     ]+  // top
								u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
            }
        if(myid < numprocs - 1)
		{
			//El ultimo solo recibe
			//printf("ULTIMO: Sending %i doubles to %i at %i\n ", by, myid+1,(sizex-2)*sizey+jj*by);
			MPI_Send(&u[ 1+sizey*(sizex-2)+jj*by ], by, MPI_DOUBLE, myid+1, 0, MPI_COMM_WORLD);	
		}
    }


    return sum;
}

