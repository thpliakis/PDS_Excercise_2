#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "cblas.h"
#include <time.h>
#include <math.h>
#include "mpi.h"

#define MASTER 0
#define FROM_MASTER 1  
#define FROM_WORKER 2

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
void printMatrix(double *d,int n,int m);
void printRes(knnresult k);

knnresult distrAllkNN(double *X, int n, int d, int k);
knnresult  mergeRes(knnresult currentKnn,knnresult prev);
void correctIdxs(knnresult *currentKnn,int offs,int rows);

int main(int argc, char *argv[]){

    int m = 3;
    int n = 13;
    int d = 3;
    int k =6;
    srand(time(NULL));
    //double  * X = (double * ) malloc( n*d * sizeof(double) );
    //double  * Y  = (double * ) malloc( m*d * sizeof(double) );
    //for (int i=0;i<n*d;i++)
    // X[i]= ( (double) (rand()%10000) ) / 50.0;
    //for (int i=0;i<m*d;i++)
    //Y[i]= ( (double) (rand()%100) ) / 50.0;

    double X[39] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0,4.0,6.0,2.0,6.0,8.0,1.0,5.0,7.0,8.0,1.0,2.0,3.0,3.0,4.0,1.0,7.0,2.0,4.0,3.0,2.0,1.0,1.0,3.0,2.0,4.0,5.0,2.0,2.0,2.0};
    //double Y[6] = {1.0,1.0,3.0,2.0,4.0,5.0};
    //double *D = distMatrix(X,Y,n,m,d);

    knnresult knnfinal = distrAllkNN(X,n,d,k);
    //knnresult test = kNN(X,X,n,n,d,k);
    //printMatrix(knnfinal.ndist,n,k);

    MPI_Finalize();
    return 0;
}

knnresult distrAllkNN(double* X, int n,int d,int k){
    int pid;
    int nproc;
    int rowchunk,extraRows,procType,rows,offset,numworkers,dest,source;

    knnresult knnfinal;
    //Initialize
    knnfinal.ndist = (double *)malloc(n*(k)*sizeof(double));
    knnfinal.nidx = (int *)malloc(n*(k)*sizeof(int));
    knnfinal.m = n;
    knnfinal.k = k;
    MPI_Request reqs[4];   // required variable for non-blocking calls
    MPI_Status stats[4];   // required variable for Waitall routine
    MPI_Status status;
    MPI_Init(0, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    numworkers = nproc -1 ;

    //Give master and workers different work 
    if (pid == MASTER){
        struct timeval startwtime, endwtime;
        double seq_time;
        gettimeofday( &startwtime, NULL );
        procType = FROM_MASTER;
        rowchunk = n/numworkers;
        extraRows = n%numworkers;
        offset=0;

        //Broke X to chunks and sent them to the workers
        for (dest=1; dest<=numworkers; dest++){
            rows = (dest <= extraRows) ? rowchunk+1 : rowchunk;
            MPI_Send(&offset, 1, MPI_INT, dest, procType, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, procType, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        //Wait for the workers to finish and gather the result
        procType = FROM_WORKER;
        int count =0;
        for (int i=1; i<=numworkers; i++){
            source = i;
            rows = (source <= extraRows) ? rowchunk+1 : rowchunk;
            MPI_Recv(&knnfinal.nidx[count], rows*(k) , MPI_INT, source, 5, MPI_COMM_WORLD, &status);
            MPI_Recv(&knnfinal.ndist[count], rows*(k) , MPI_DOUBLE, source, 6, MPI_COMM_WORLD, &status);
            count += rows*(k);
        }
        gettimeofday( &endwtime, NULL );
        seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
        printf("Asychronous time is %f secconds for n=%d, d=%d and k=%d and nporc=%d\n",seq_time,n,d,k,nproc );
        //printRes(knnfinal);
        printMatrix(knnfinal.ndist,n,k);

        return knnfinal;
    }
    if(pid > MASTER){
        //printf("hello from procces %d\n",pid );
        //Calculate next proc to pass data and prev proc to receive data
        int coffset,crows;
        int  prev = pid-1;
        int next = pid+1;
        //Where matrices will be saved
        double *Xi = &X[(offset)*d];
        double *Y =  &X[(offset)*d];
        if (pid == 1) 
            prev = nproc - 1;
        if (pid == (nproc - 1)) 
            next = 1;
        //Initial row and offset receive from master
        procType = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, procType, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, procType, MPI_COMM_WORLD, &status);

        knnresult currentKnn;
        //Asynchronous isend and irecv in order to do calculations simultaneously
        procType = FROM_WORKER;
        MPI_Isend(&offset,1,MPI_INT,next,3,MPI_COMM_WORLD,&reqs[0]);
        MPI_Isend(&rows,1,MPI_INT,next,4,MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(&coffset, 1, MPI_INT, prev, 3, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&crows, 1, MPI_INT, prev, 4, MPI_COMM_WORLD, &reqs[3]);

        knnresult previusKnn = kNN(Xi,Y,rows,rows,d,k);
        correctIdxs(&previusKnn,offset,rows);
        MPI_Waitall(4, reqs, stats);  //Wait all procceces

        for(int i =1;i<=numworkers-1;i++){
            Xi = &X[coffset*d];
            int count1,count2;  // Temp values to asynchronously receive new offset and rows
            //Asynchronous isend and irecv in order to do calculations simultaneously
            MPI_Isend(&coffset,1,MPI_INT,next,3,MPI_COMM_WORLD,&reqs[0]);
            MPI_Isend(&crows,1,MPI_INT,next,4,MPI_COMM_WORLD,&reqs[1]);
            MPI_Irecv(&count1, 1, MPI_INT, prev, 3, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&count2, 1, MPI_INT, prev, 4, MPI_COMM_WORLD, &reqs[3]);

            currentKnn = kNN(Xi,Y,crows,rows,d,k);
            correctIdxs(&currentKnn,coffset,crows);
            previusKnn = mergeRes(currentKnn,previusKnn);

            MPI_Waitall(4, reqs, stats); //Wait all procceces
            coffset = count1;
            crows = count2;
        }
        printf("Hello from procces %d with rows %d\n",pid,previusKnn.m);
        printMatrix(previusKnn.ndist,previusKnn.m,k);

        MPI_Send(&(previusKnn.nidx[0]), (k)*rows ,MPI_INT, MASTER, 5, MPI_COMM_WORLD);
        MPI_Send(&(previusKnn.ndist[0]), (k)*rows ,MPI_DOUBLE, MASTER, 6, MPI_COMM_WORLD);
    }
}


void correctIdxs(knnresult *currentKnn,int offset,int rows){
    int k = currentKnn->k;
    int m = currentKnn->m;
    for (int i =0;i<m;i++ )
        for(int j = 0;j<k;j++)
            currentKnn->nidx[i*k + j] += offset;
}

knnresult  mergeRes(knnresult currentKnn,knnresult previus){
    knnresult tempres;
    int k = currentKnn.k;
    int m = currentKnn.m;
    int icur=0;
    int iprev=0;
    tempres.ndist = (double*)malloc(m*k*sizeof(double));
    tempres.nidx = (int*)malloc(m*k*sizeof(int));
    tempres.m = m;
    tempres.k =k;  

    for(int i =0;i<m;i++){
        icur =0;
        iprev =0;
        for(int j=0;j<k;j++){
            if (     previus.ndist[i*k+iprev] <= currentKnn.ndist[i*k+icur]     ){
                tempres.ndist[i*k + j] = previus.ndist[i*k + iprev];
                tempres.nidx[i*k  + j] = previus.nidx[i*k + iprev];
                iprev++;
            }else{
                tempres.ndist[i*k + j] = currentKnn.ndist[i*k + icur];
                tempres.nidx[i*k  + j] = currentKnn.nidx[i*k + icur];
                icur++;
            }
        }
    }

    return tempres;
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k){
    knnresult knnRes;
    double *D;
    int *idx,temp=0;

    printf("k = %d\n n = %d\n",k,n);
    if(k<=0 || k > n){
        printf("k must be > 0 and <  than %d  \n Goodbye now ...\n",n);
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

    printf(" \n");

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
    //printMatrix(knnRes.ndist,m,k);
    //  printf("\n");

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
        printf("Y point %d has : \n",i);
        for(int j = 0 ;j< R.k;j++)
            printf("%d neighboor with %f distance and indx %d\n",j+1,R.ndist[i*R.k+j],R.nidx[i*R.k+j]);
    }
}

void printMatrix(double *d,int n,int m){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            printf("%f  ",d[i*m+j] );
        }
        printf("\n");
    }
}
