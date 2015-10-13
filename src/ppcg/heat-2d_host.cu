#include <assert.h>
#include <stdio.h>
#include "heat-2d_kernel.hu"
/*
 * Discretized 2D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 *
 * Irshad Pananilath: irshad@csa.iisc.ernet.in
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#ifdef USE_LIKWID
#include<likwid.h>
#endif


/*
 * N is the number of points
 * T is the number of timesteps
 */
#ifdef HAS_DECLS
#include "decls.h"
#else
#define N 4000L
//#define T 1000L
#endif

#define NUM_FP_OPS 10

/* Define our arrays */
double A[2][N+2][N+2];
double total=0; double sum_err_sqr=0;
int chtotal=0;
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y) {
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;

        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }

    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;

        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    return x->tv_sec < y->tv_sec;
}

int main(int argc, char * argv[]) {
    long int i, j;
    const int BASE = 1024;

    // for timekeeping
    struct timeval start, end, result;
    double tdiff = 0.0;
    
    int T;

    printf("Please enter number of timesteps = \n");
    scanf("%d", &T);

    printf("Number of points = %ld\t|Number of timesteps = %ld\t", N*N, T);

    /* Initialization */
    srand(42); // seed with a constant value to verify results

    for (i = 0; i <= N+1; i++) {
        for (j = 0; j <= N+1; j++) {
            A[0][i][j] = 1.0 * (rand() % BASE);
        }
    }

#ifdef USE_LIKWID
#pragma omp parallel
{
LIKWID_MARKER_START("Compute_omp");
}
#endif

    if (T >= 1)
      {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

        double *dev_A;
        
        cudaCheckReturn(cudaMalloc((void **) &dev_A, (2) * (4002) * (4002) * sizeof(double)));
        
        cudaCheckReturn(cudaMemcpy(dev_A, A, (2) * (4002) * (4002) * sizeof(double), cudaMemcpyHostToDevice));

#ifdef TIME
        gettimeofday(&start, 0);
#endif
        
        #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
        for (int c0 = 0; c0 < T; c0 += 1)
          {
            dim3 k0_dimBlock(16, 32);
            dim3 k0_dimGrid(126, 126);
            kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, T, c0);
            cudaCheckKernel();
            cudaCheckReturn(cudaDeviceSynchronize());
          }

#ifdef TIME
        gettimeofday(&end, 0);

        timeval_subtract(&result, &end, &start);
        tdiff += (double)(result.tv_sec + result.tv_usec * 1.0e-6);
#endif
          
        cudaCheckReturn(cudaMemcpy(A, dev_A, (2) * (4002) * (4002) * sizeof(double), cudaMemcpyDeviceToHost));
        
        cudaCheckReturn(cudaFree(dev_A));
      }

#ifdef TIME
    printf("|Time taken =  %7.5lfms\t", tdiff * 1.0e3);
    printf("|MFLOPS =  %f\t", ((((double)NUM_FP_OPS * N *N *  T) / tdiff) / 1000000L));
#endif

#ifdef USE_LIKWID
#pragma omp parallel
{
LIKWID_MARKER_STOP("Compute_omp");
}
#endif


#ifdef VERIFY
    for (i = 1; i < N+1; i++) {
        for (j = 1; j < N+1; j++) {
            total+= A[T%2][i][j] ;
        }
    }
    printf("|sum: %e\t", total);
    for (i = 1; i < N+1; i++) {
        for (j = 1; j < N+1; j++) {
            sum_err_sqr += (A[T%2][i][j] - (total/N))*(A[T%2][i][j] - (total/N));
        }
    }
    printf("|rms(A) = %7.2f\t", sqrt(sum_err_sqr));
    for (i = 1; i < N+1; i++) {
        for (j = 1; j < N+1; j++) {
            chtotal += ((char *)A[T%2][i])[j];
        }
    }
    printf("|sum(rep(A)) = %d\n", chtotal);
#endif
    return 0;
}

// icc -O3 -fp-model precise heat_1d_np.c -o op-heat-1d-np -lm
// /* @ begin PrimeTile (num_tiling_levels=1; first_depth=1; last_depth=-1; boundary_tiling_level=-1;) @*/
// /* @ begin PrimeRegTile (scalar_replacement=0; T1t3=8; T1t4=8; ) @*/
// /* @ end @*/
