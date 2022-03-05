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
        /*Gaussian elimination*/
        for (k = 0; k < size - 1; ++k)
        {
            /*Pivoting*/
            temp = 0;
            for (i = k, j = 0; i < size; ++i)
                if (temp < A[index[i]][k] * A[index[i]][k])
                {
                    temp = A[index[i]][k] * A[index[i]][k];
                    j = i;
                }
            if (j != k) /*swap*/
            {
                i = index[j];
                index[j] = index[k];
                index[k] = i;
            }
            /*calculating*/
            for (i = k + 1; i < size; ++i)
            {
                temp = A[index[i]][k] / A[index[k]][k];
                for (j = k; j < size + 1; ++j)
                    A[index[i]][j] -= A[index[k]][j] * temp;
            }
        }
        /*Jordan elimination*/
        for (k = size - 1; k > 0; --k)
        {
            for (i = k - 1; i >= 0; --i)
            {
                temp = A[index[i]][k] / A[index[k]][k];
                A[index[i]][k] -= temp * A[index[k]][k];
                A[index[i]][size] -= temp * A[index[k]][size];
            }
        }
        /*solution*/
        for (k = 0; k < size; ++k)
            X[k] = A[index[k]][size] / A[index[k]][k];
    }
    GET_TIME(end);
    Lab3SaveOutput(X, size, end - start);
}