#include <iostream>
#include <chrono>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

inline  void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}



__global__ void convolution(float *image, float *kernel, int n , int m, int k,float *out_image)
{
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        //printf("row=%d\tcol=%d\n",row,col);
        if (row >= m || col >= n)
        return;


        float add=0;
        int mask= k/2;

        for(int i=-mask; i<=mask; i++)
        {
                for(int j=-mask; j<=mask; j++)
                {
                        if((row+i)>=0 && (row+i)<m && (col+j)>=0 && (col+j)<n)
                        {
                                add+= image[(row+i) *n + col+j] * kernel[(mask+i)*k + (mask+j)];
                        }


                }
        }

        out_image[(row*n)+col]= add;

}

__global__ void matrixmul(float* feature_matrix, float* fullyconnected, float* matrix_output,int feature_rows,int feature_columns,int fully_rows,int fully_columns, int output_rows,int output_columns,int pooling_pixels )
{

        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row > output_rows || col > output_columns)
         return;

        for (int e=0; e<pooling_pixels; ++e)
        {
                Cvalue += feature_matrix[row * feature_rows + e] *fullyconnected[e *fully_columns + col];
        }

        matrix_output[row * output_rows +col] = Cvalue;

}

/**************************************************Rectified LInear UNit******************************************************************************************************/



__global__ void Relu(float* image,int n,int m, float* output_image)
{
        int max=25;
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (row >= m || col >= n || row<0 || col<0)
        return;

        if(image[row*n+col] >max)
        {
                //printf("Hey");
                output_image[row*n+col]=max;
        }

}

/*******************************************************Pooling Layer*******************************************************************************/
__global__ void Pool(float* image,int n,int m, float* output_image)
{
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        int row1=row/2;
        int col1=col/2;
        int tot_rows=(m/2)+1;
        int tot_cols=(n/2)+1;

        if (row >= m || col >= n || row<0 || col<0)
        return;

        int max1=0;
        int max2=0;
        int max3=0;
if((row%2==0) && (col%2==0))
{
        //printf("Row and col are %d %d\n",row,col);
        if (row+1<m && col+1<n)
        {
                max1 = image[row*n+col]>=image[row*n+(col+1)]? image[row*n+col]:image[row*n+(col+1)];
                max2 = image[(row+1)*n+col]>=image[(row+1)*n+col+1]? image[(row+1)*n+col]:image[(row+1)*n+col+1];
                max3 = max1>=max2? max1:max2;
        }
        else if (row+1==m && col+1==n)
        {
                max3 = image[row*n+col];
        }
        else if (row+1==m)
        {
                max3 = image[(row*n)+col]>=image[(row*n)+col+1]? image[(row*n)+col]:image[(row*n)+col+1];
        }
        else if (col+1==n)
        {
                max3 = image[(row*n)+col]>=image[(row+1)*n+col]?image[(row*n)+col]:image[(row+1)*n+col];
        }

        output_image[(row1*tot_cols)+col1]= max3;
        //printf("Row1 and col1 are %d %d indexing is %d data is %f\n",row1,col1,(row1*tot_cols)+col1,output_image[(row1*tot_cols)+col1]);

}

}

/***************************************************************Main********************************************************************/



int main (int argc, char* argv[]) {


  if(argc!=4)
  {
        printf("Not enough arguments");
        return 0;
  }

  int m = atoi(argv[1]);
  printf("M is  %d\n",m);

  int n = atoi(argv[2]); //TODO: atoi is an unsafe function
  printf("N is  %d\n",n);


  int k = atoi(argv[3]);
  printf("The kernel size is %d\n",k);

  int total_pixels=m*n;
  int m_pool=((m-2)/2)+2;
  int n_pool=((n-2)/2)+2;
  int pool_pixels=m_pool*n_pool;
  printf("pool_pixels are %d\n", pool_pixels);

  int i,j;
  int height=1;

  float *image= new float [total_pixels];
  float *G_image= new float[total_pixels];
  float *B_image= new float[total_pixels];
  float *kernel= new float[k*k];
  float *output_image= new float[total_pixels];
  float *G_output_image= new float[total_pixels];
  float *B_output_image= new float[total_pixels];
  float *relu_output= new float[total_pixels];
  float *pool_output= new float[pool_pixels];
  float *feature_matrix= new float[pool_pixels];
  float *fullyconnected= new float[pool_pixels *100];
  float *matrix_output= new float[100];

  
  float *d_feature_matrix;
  float *d_fullyconnected;
  float *d_matrix_output;
  float *d_image;
  float *d_G_image;
  float *d_B_image;
  float *d_outimage;
  float *d_G_outimage;
  float *d_B_outimage;
  float *d_kernel;
  float *d_reluimage;
  float *d_reluoutput;
  float *d_poolimage;
  float *d_pooloutput;

  for(i=0; i < total_pixels; i++)
  {
        image[i]=1.0;
        G_image[i]=1.0;
        B_image[i]=1.0;
        output_image[i]=0;
        G_output_image[i]=0;
        B_output_image[i]=0;
  }


  for(i = 0; i < k*k; i++)
  {

        kernel[i]= 1;
  }

  for(i=0;i<100;i++)
  {
        matrix_output[i]=2.0;
  }
  
   for(i=0;i<(pooling_pixels *100);i++)
  {
        fullyconnected[i]=1.0;
  }

   for(i=0;i<pooling_pixels;i++)
   {
        feature_matrix[i]=1.0;
   }



 std::chrono::time_point<std::chrono::system_clock> begin, end, begin1, end1,begin2,end2,begin3,end3;

  HANDLE_ERROR(cudaMalloc((void**)&d_image, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_G_image, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_B_image, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_kernel, (k*k)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_outimage, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_G_outimage, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_B_outimage, (total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_reluimage,(total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_reluoutput,(total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_poolimage,(total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_pooloutput,(total_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_feature_matrix,(pool_pixels)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_fullyconnected,(pool_pixels*100)*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&d_matrix_output,(100)*sizeof(float)));


  HANDLE_ERROR(cudaMemcpy(d_outimage,output_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_G_outimage,G_output_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_B_outimage,B_output_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_image,image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_G_image,G_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_B_image,B_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_kernel,kernel,(k*k)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_feature_matrix,feature_matrix,(pool_pixels)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_fullyconnected,fullyconnected,(pool_pixels*100)*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_matrix_output,matrix_output,(100)*sizeof(float),cudaMemcpyHostToDevice));


  const dim3 blocksize(16,16);

  const dim3 gridsize(n/blocksize.y +1,m/blocksize.x+1);
  printf("gridsize.x=%d, gridsize.y=%d\n",gridsize.x,gridsize.y);
  begin = std::chrono::system_clock::now();

   /* cudaSetDevice(0);
  convolution<<<gridsize,blocksize>>>(d_image,d_kernel,n,m,k,d_outimage);

  cudaSetDevice(1);
  convolution<<<gridsize,blocksize>>>(d_G_image,d_kernel,n,m,k,d_G_outimage);

  cudaSetDevice(0);
  cudaDeviceSynchronize();

  cudaSetDevice(0);
  convolution<<<gridsize,blocksize>>>(d_B_image,d_kernel,n,m,k,d_B_outimage);

  cudaSetDevice(1);
  cudaDeviceSynchronize();

  cudaSetDevice(0);
  cudaDeviceSynchronize();*/

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> totaltime = (end-begin);

  std::cout<<" For array size " << n <<" The time required for convolution is "<<(totaltime.count())<<std::endl;



  HANDLE_ERROR(cudaMemcpy(output_image, d_outimage, (total_pixels)*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(G_output_image, d_G_outimage, (total_pixels)*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(B_output_image, d_B_outimage, (total_pixels)*sizeof(float),cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
  
  for(i=0;i<total_pixels;i++)
  {
		output_image[i]+=G_output_image[i]+B_output_image[i];
  }
/*****************************Second Kernel Launch****************************************************/

  HANDLE_ERROR(cudaMemcpy(d_reluimage,output_image,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));

   begin1 = std::chrono::system_clock::now();
   Relu<<<gridsize,blocksize>>>(d_reluimage,n,m,d_outimage);
   cudaDeviceSynchronize();
   end1 = std::chrono::system_clock::now();
   std::chrono::duration<double> totaltime1 = (end1-begin1);

    std::cout<<" For array size " << n <<" The time required for Relu layer is "<<(totaltime1.count())<<std::endl;

  HANDLE_ERROR(cudaMemcpy(relu_output,d_outimage, (total_pixels)*sizeof(float),cudaMemcpyDeviceToHost));

/******************************************************************************************************/

	cudaDeviceSynchronize();

/*****************************Third Kernel Launch****************************************************/

HANDLE_ERROR(cudaMemcpy(d_poolimage,relu_output,(total_pixels)*sizeof(float),cudaMemcpyHostToDevice));

 begin2 = std::chrono::system_clock::now();
 Pool<<<gridsize,blocksize>>>(d_poolimage,n,m,d_pooloutput);
 cudaDeviceSynchronize();
 end2 = std::chrono::system_clock::now();

 std::chrono::duration<double> totaltime2 = (end2-begin2);

 std::cout<<" For array size " << n <<" The time required for Pool layer is "<<(totaltime2.count())<<std::endl;


 HANDLE_ERROR(cudaMemcpy(pool_output,d_pooloutput, (pool_pixels)*sizeof(float),cudaMemcpyDeviceToHost));

/************************************************************************************************************/
cudaDeviceSynchronize();


/******************************************************************************Fully_Connected Kernel********************************************************************************/
 begin3 = std::chrono::system_clock::now();

 matrixmul<<<1,110>>>(d_feature_matrix,d_fullyconnected,d_matrix_output,height,pool_pixels,pool_pixels,100,0,100,pool_pixels);
 cudaDeviceSynchronize();
 end3 = std::chrono::system_clock::now();

 std::chrono::duration<double> totaltime3 = (end3-begin3);

 std::cout<<" For array size " << n <<" The time required for Matrix Multiplication layer is "<<(totaltime3.count())<<std::endl;

  HANDLE_ERROR(cudaMemcpy(matrix_output, d_matrix_output, (100)*sizeof(float),cudaMemcpyDeviceToHost));
/************************************************************************************************************************************************************************************/

HANDLE_ERROR(cudaFree(d_image));
HANDLE_ERROR(cudaFree(d_G_image));
HANDLE_ERROR(cudaFree(d_B_image));
HANDLE_ERROR(cudaFree(d_kernel));
HANDLE_ERROR(cudaFree(d_outimage));
HANDLE_ERROR(cudaFree(d_G_outimage));
HANDLE_ERROR(cudaFree(d_B_outimage));
HANDLE_ERROR(cudaFree(d_reluimage));
HANDLE_ERROR(cudaFree(d_reluoutput));
HANDLE_ERROR(cudaFree(d_poolimage));
HANDLE_ERROR(cudaFree(d_pooloutput));
HANDLE_ERROR(cudaFree(d_feature_matrix));
HANDLE_ERROR(cudaFree(d_fullyconnected));
HANDLE_ERROR(cudaFree(d_matrix_output));
free(image);
free(G_image);
free(B_image);
free(kernel);
free(output_image);
free(G_output_image);
free(B_output_image);

}



		
		