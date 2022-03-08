#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "Lab3IO.h"
#include "timer.h"

int main(int argc, char *argv[])
{

    double **A;
    int size;

    double *X;
    int *index;
    int i, j, k;
    double temp;

    double start;
    double end;


    // Load in the matrix
    int numThreads = strtol(argv[1], NULL, 10);

    Lab3LoadInput(&A, &size);

    /*Calculate the solution by serial code*/
    X = CreateVec(size);
    index = malloc(size * sizeof(int));
    for (i = 0; i < size; ++i)
        index[i] = i;

    // Start recording processing time
    GET_TIME(start);
    if (size == 1)
        X[0] = A[0][1] / A[0][0];
    else
    {
        // omp_set_dynamic(1);
        /*Gaussian elimination*/
        #pragma omp parallel num_threads(numThreads)
        {
            #pragma omp single nowait
            {
                for (k = 0; k < size - 1; ++k)
                {
                    /*Pivoting*/
                    temp = 0;
                    j = 0;
                    for (i = k; i < size; ++i)
                        #pragma omp task firstprivate(i) shared(temp,A,k,size,index,j)
                        {
                            if (temp < A[index[i]][k] * A[index[i]][k])
                            {
                                #pragma omp critical
                                if (temp < A[index[i]][k] * A[index[i]][k])
                                {
                                    temp = A[index[i]][k] * A[index[i]][k];
                                    j = i;
                                }
                            }
                        }
                    #pragma omp taskwait
                    if (j != k) /*swap*/
                    {
                        i = index[j];
                        index[j] = index[k];
                        index[k] = i;
                    }
                    /*calculating*/
                    for (i = k + 1; i < size; ++i)
                    {
                        #pragma omp task firstprivate(i,temp,j) shared(A,k,index,size)
                        {
                            temp = A[index[i]][k] / A[index[k]][k];
                            for (j = k; j < size + 1; ++j)
                                A[index[i]][j] -= A[index[k]][j] * temp;

                        }
                    }
                    #pragma omp taskwait
                }

                /*Jordan elimination*/
                for (k = size - 1; k > 0; --k)
                {
                    for (i = k - 1; i >= 0; --i)
                    {
                        #pragma omp task firstprivate(temp,i) shared(A,k,size,index)
                        {
                            temp = A[index[i]][k] / A[index[k]][k];
                            A[index[i]][k] -= temp * A[index[k]][k];
                            A[index[i]][size] -= temp * A[index[k]][size];
                        }
                    }
                    #pragma omp taskwait
                }
                /*solution*/
                for (k = 0; k < size; ++k) {
                    #pragma omp task firstprivate(k) shared(index,A,size,X)
                    {
                        X[k] = A[index[k]][size] / A[index[k]][k];
                    }
                }
                #pragma omp taskwait
            }
        }
    }
    GET_TIME(end);
    Lab3SaveOutput(X, size, end - start);
    printf("time: %f\n", end-start);
    DestroyVec(X);
    DestroyMat(A, size);
    free(index);
}