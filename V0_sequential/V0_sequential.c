#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "cblas.h"
#include <time.h>
#include <math.h>

typedef struct knnresult{
    int    *nidx;      //!<Indices (0-based) of nearest neighbors [m-by-k]
    double *ndist;     //!< Distance of nearest neighbors         [m-by-k]
    int     m;         //!< Number of query points                [scalar]
    int     k;         //!< Number of nearest neighbors           [scalar]
} knnresult;

knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
int partition(double *l,int *idx, int left, int right);
double qselect(double *l,int * idx, int left, int right, int k);
void swap(double *v,int *idx, int k, int l);
void quickSort(double *l ,int *idx, int left, int right);
void D_calc(double *X, double *Y, int n, int m, int d, double *D);
//Function to print matrices
void printMatrix(double *d,int n,int m);
//Funcrion to print k.ndist
void printRes(knnresult k);

int main(int argc, char *argv[]){

    //initiallize some data
    double X[12] = {6.0, 4.0,3.0,2.0,6.0,2.0,9.0,5.0,2.0,7.0,1.0,3.0};
    double Y[9] = {5.0,10.0,4.0,1.0,5.0,10.0,10.0,5.0,4.0};
    int n,m,d,k;

    n=4;
    m=3;
    d=3;
    k=3;

    struct timeval startwtime, endwtime;
    double seq_time;
    gettimeofday( &startwtime, NULL );
    knnresult knn = kNN(X,Y,n,m,d,k);
    gettimeofday( &endwtime, NULL );
    seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
    printf("Serial method time is %f secconds.\n",seq_time );
    printRes(knn);
    printf("ndist matrix : \n");
    printMatrix(knn.ndist,knn.m,knn.k);
    return 0;
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k){
    knnresult knnRes;
    double *D;
    int *idx,temp=0;

    if(k<=0 || k > n){
        printf("k must be > 0 and < than %d  \n Goodbye now ...\n",n);
        exit(1);
    }

    //initiallize
    knnRes.nidx = (int*)malloc(m*n*sizeof(int));
    knnRes.ndist = (double*)malloc(n*m*sizeof(double));
    knnRes.m = m;
    knnRes.k = k;

    D = (double *) malloc(n * m * sizeof(double));
    double  *Dtrans = (double *) malloc(m*n*sizeof(double));
    idx = (int *) malloc(n * m * sizeof(int));
    for(int j=0; j<m; j++){
        for (int i=0; i<n; i++){
            idx[i+j*n] = i;
            //printf("%d  ",idx[i+j*n]);
        }
        //printf(" \n");
    }

    //calculate distance matrix
    D_calc(X,Y,n,m,d,D);
    //printMatrix(D,n,m);
   
    //calculate transpose Distance matrix
    for(int j=0; j<m; j++)
        for(int i=0; i<n; i++)
            Dtrans[temp++] = D[i*m+j];
    //printMatrix(Dtrans,m,n);

    //printf(" \n");

    //fill ndist and nidx with knn
    temp=0;
    for(int j=0; j<m; j++){
        qselect(&Dtrans[j*n],&idx[j*n],0,n-1,k);
        quickSort(&Dtrans[j*n],&idx[j*n],0,k-1);
        for(int i=0; i<k; i++){
            knnRes.nidx[temp]  = idx[j*n + i];
            if(Dtrans[j*n + i] < 10e-8) Dtrans[j*n + i] =0;
            knnRes.ndist[temp++] = sqrt(Dtrans[j*n + i]);
        }
    }

    free(D);
    free(idx);

    return knnRes;
}

void D_calc(double *X, double *Y, int n, int m, int d, double *D){
    //printf("X = \n");
    //printMatrix(X,n,d);
    //printf("Y = \n");
    //printMatrix(Y,m,d);

    for(int i =0 ;i<n;i++){
        double x =  cblas_dnrm2(d,&X[i*d],1);
        //printf("x = %f\n",x*x);
        for(int j =0;j<m;j++){
            double y =  cblas_dnrm2(d,&Y[j*d],1);
            //printf("y = %f\n",y*y);
            D[i*m+j] = x*x + y*y;
        }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,m,d,-2,X,d,Y,d,1,D,m);
    //printMatrix(D,n,m);

}

/* swap -- swap elements k and l of vector v */
void swap(double *v,int *idx, int k, int l) {
    double temp = v[k];
    v[k] = v[l];
    v[l] = temp;
    int temp2 = idx[k];
    idx[k] = idx[l];
    idx[l] = temp2;
}

int partition(double *l,int *idx ,int left, int right){
    double pivotValue = l[right];
    int storeIndex;

    storeIndex = left;
    for(int i=left; i<right; i++){
        if(l[i] < pivotValue){
            swap(l,idx,storeIndex,i);
            storeIndex++;
        }
    }
    swap(l,idx,storeIndex,right);
    return storeIndex;
}

double qselect(double *l,int *idx, int left, int right, int k){
    int pivotIndex = partition(l,idx,left,right);
    if(pivotIndex == k) return l[pivotIndex];
    if(k < pivotIndex){
        return qselect(l,idx,left,pivotIndex-1,k);
    }else {
        return qselect(l,idx,pivotIndex+1,right,k);
    }
}

void quickSort(double *l ,int *idx, int left, int right){
    if (left < right){
        int pi = partition(l,idx, left, right);
        quickSort(l,idx, left, pi - 1);
        quickSort(l,idx, pi + 1, right);
    }
}

void printRes(knnresult R){
    for(int i = 0;i<R.m;i++){
        printf(" Y point %d has : \n",i);
        for(int j = 0 ;j< R.k;j++)
            printf("%d neighboor with %f distance and indx %d\n",j+1,R.ndist[i*R.k+j],R.nidx[i*R.k+j]);
    }
}

void printMatrix(double *d,int n,int m){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            printf("%f  ",d[i*m+j] );
        }
        printf(" \n");
    }
}
