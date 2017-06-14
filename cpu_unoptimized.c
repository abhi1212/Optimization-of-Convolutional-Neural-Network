#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <math.h>
using namespace std;


/******************************************************Function Declaration*********************************************************************************************************/

void initialize (uint32_t m, uint32_t n, uint32_t k, int p_m, int p_n, float **R_image, float **G_image, float **B_image,
float **R_kernel, float **G_kernel, float **B_kernel, float **output, float **pooling_output, float **feature_matrix, float **fullyconnected, float **multiplied_result);


void convolution (float **output, float **R_image, float **G_image, float **B_image, float **R_kernel, float **G_kernel, float **B_kernel, int kernelcenterX, int kernelcenterY, uint32_t m, uint32_t n, uint32_t k);

void rectifiedLinearUnit (float **output, int m, int n, int max);
void poolingLayer (float **output, float **pooling_output, int m, int n, int p_m, int p_n);

void matrixMultiplication (float **feature_matrix, float **fullyconnected, float **multiplied_result, float **pooling_output, int p_m, int p_n, int key);


/************************************************************************************************************************************************************************************/

/***********************************************************************************************************************************************************************************
Function Name- Main()

Function Purpose- Accept the Size of the Image and Kernel and call the necessary functions

***********************************************************************************************************************************************************************************
int main(int argc, char** argv)
 {

    if (argc!= 4)
        {
        cout<<"Not enough parameters";
        return 0;
        }

        int i=0;
        int j=0;

    uint32_t m= atoi(argv[1]);
    cout<<"m is "<<m<<endl<<endl;

    uint32_t n = atoi(argv[2]);
    cout<<"n is "<< n << endl<<endl;

    uint32_t k =atoi(argv[3]);
    cout<<"K is"<<k<<endl<<endl;


    int kernelcenterX= k/2;
    int kernelcenterY= k/2;
    int max=20;
    int p_m= floor(m/2);
    int p_n= floor(n/2);
    float** R_image= new float*[m];
    float** G_image= new float*[m];
    float** B_image= new float*[m];


    float** R_kernel= new float*[k];
    float** G_kernel= new float*[k];
    float** B_kernel= new float*[k];
    float** output= new float*[m];
    float** pooling_output= new float*[(p_m)];
    float** feature_matrix= new float*[1];
    float** fullyconnected= new float*[p_m*p_n];
    float** multiplied_result = new float*[1];
    int key=0;

    initialize (m, n, k, p_m, p_n, R_image, G_image, B_image,
                                        R_kernel, G_kernel, B_kernel, output, pooling_output, feature_matrix,
                                        fullyconnected, multiplied_result);
                                
								 convolution (output, R_image, G_image, B_image, R_kernel, G_kernel, B_kernel, kernelcenterX, kernelcenterY, m, n, k);


    rectifiedLinearUnit (output, m, n, max);

    poolingLayer (output, pooling_output, m, n, p_m, p_n);

    matrixMultiplication (feature_matrix, fullyconnected, multiplied_result, pooling_output, p_m, p_n, key);

}


/***********************************************************************************************************************************************************************************
 * Function Name- Initialize()
 *
 * Function Purpose- Seperate the Image into 3 channels and initiliaze Kernels properly
 *
 * *************************************************************************************************************************************************************************************/

void initialize (uint32_t m, uint32_t n, uint32_t k, int p_m, int p_n, float **R_image, float **G_image, float **B_image,float **R_kernel, float **G_kernel, float **B_kernel, float **output, float **pooling_output, float **feature_matrix, float **fullyconnected, float **multiplied_result)
{
        int i, j;
        for( i=0;i<m;i++)
        {
                R_image[i]= new float[n];
                G_image[i]= new float[n];
                B_image[i]= new float[n];
                output[i]= new float[n];
                pooling_output[i]= new float[(p_n)];
        }




        for( i=0;i<k;i++)
        {
                R_kernel[i]= new float[k];
                G_kernel[i]= new float[k];
                B_kernel[i]= new float[k];
        }



        for(i=0; i<m; i++)
        {
                for( j=0;j<n;j++)
                {
                        R_image[i][j]= 1;
                        G_image[i][j]= 1;
                        B_image[i][j]= 1;
                        output[i][j]= 0;
                }
        }



        for(i=0; i<k; i++)
        {
                for(j=0;j<k;j++)
                 {
                        R_kernel[i][j]= 1;
                        G_kernel[i][j]= 1;
                        B_kernel[i][j]= 1;
                 }
        }

  feature_matrix[0]= new float[p_m*p_n];
        multiplied_result[0] = new float[100];

        for(i=0;i<(p_m*p_n);i++)
    {
        fullyconnected[i]= new float[100];

    }


}

/***********************************************************************************************************************************************************************************
 *  * Function Name- Convolution()
 *   *
 *    * Function Purpose- Performs Convolution on Images
 *     *
 * *************************************************************************************************************************************************************************************/

void convolution (float **output, float **R_image, float **G_image, float **B_image, float **R_kernel, float **G_kernel, float **B_kernel,
           int kernelcenterX, int kernelcenterY, uint32_t m, uint32_t n, uint32_t k)
{

        int i, j;
        auto start_time = chrono::high_resolution_clock::now();

        for(int rows=0;rows<m;rows++)
        {
                for(int columns=0;columns<n;columns++)
                {
                        for(int krows=0;krows<k;krows++)
                        {
                                int mm = k - 1 - krows;
                                for(int kcolumns=0; kcolumns<k;kcolumns++)
                                {
                                        int nn = k - 1 - kcolumns;
                                        int ii = rows + (krows - kernelcenterY);
                                        int jj = columns + (kcolumns - kernelcenterX);
                                        if(ii>=0 && ii<m && jj>=0 && jj<n)
                                        {
                                                  output[rows][columns] += (R_image[ii][jj] * R_kernel[mm][nn]) + (G_image[ii][jj] * G_kernel[mm][nn])+ (B_image[ii][jj] * B_kernel[mm][nn]);
                                        }



                                }
                        }

                }


        }

 auto end_time = chrono::high_resolution_clock::now();
        int operation_performance= (819/(k*k));
        int mem_time= ((34*m*n)/ ((2*m*n)+ (k*k)));
        int total_pixels= (3*m*n);
        //cout<<"Size of pooling layer matrix"  <<p_m<<"  "<<p_n;
        cout<<"Total_pixels are"<<total_pixels<<endl;
        cout<<"The performance for Convolution is as belows"<<endl;

        cout<<"The time in microseconds for convolution"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<" microseconds"<<endl;
        int actual= (total_pixels/ chrono::duration_cast<chrono::microseconds>(end_time - start_time).count());

        cout<<"The floptime performance for floating point operation is"<<operation_performance<<" Gigapixels"<<endl;
        cout<<"Performance for Memory Transfer "<<mem_time<<" Gigapixels/sec"<<endl;
        cout<<"The actual performance achieved is"<<actual<<" Megapixels"<<endl;

        cout<<"The matrix after convolution is"<<endl;
*/

        /* for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        cout<<" "<<output[i][j];
                }
                cout<<endl;
        }*/


}
/****************************************************************************Rectified Linear Unit********************************************************************************

Function Name- Rectified Linear Unit
Function Purpose- It is a normalizaton Layer
Perfromance Expected- The performance is expected to be Memory Bound

***********************************************************************************************************************************************************************************/

void rectifiedLinearUnit (float **output, int m, int n, int max)
{
        int i,j;
        auto start_time_relu = chrono::high_resolution_clock::now();

        for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        if(output[i][j]<0)
                        {
                                output[i][j]=0;
                        }
                        if(output[i][j]>max)
                        {
                                output[i][j]=max;
                        }
                }
        }

         auto end_time_relu = chrono::high_resolution_clock::now();

         //cout<<"The time in microseconds for relu is"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_relu - start_time_relu).count()<<" microseconds"<<endl;
         long int relu_performance=((m*n)/ chrono::duration_cast<chrono::microseconds>(end_time_relu - start_time_relu).count());
         //cout<<"The expected Performance is 50Gp/s"<<endl;
         //cout<<"Relu_performance for Relu layer is"<<relu_performance<<"Mpps"<<endl;


        /* cout<<"The final matrix after rectifiled linear unit is"<<endl;
         for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        cout<<" "<<output[i][j];
                }
                cout<<endl;
        }*/
}

/****************************************************************************Pooling Layer********************************************************************************
 *
 * Function Name- Pooling Layer
 * Function Purpose- It is a layer used in Convolutional Neural Networkcfor downsampling Image
 * Perfromance Expected- The performance is expected to be Memory Bound
 *
 * ******************************************************************************************************************************************************************************/



void poolingLayer (float **output, float **pooling_output, int m, int n, int p_m, int p_n)
{

        int i, j, max1, max2;

        auto start_time_pool = chrono::high_resolution_clock::now();
        for(i=0; i<m; i+=2)
        {
                for(j=0;j<n;j+=2)
                {
                        if (i+1<m && j+1<n)
                        {
                                max1 = output[i][j]>=output[i][j+1]? output[i][j]:output[i][j+1];
                                max2 = output[i+1][j]>=output[i+1][j+1]? output[i+1][j]:output[i+1][j+1];
                                max1 = max2>=max1? max2:max1;
                        }
                        else if (i+1==m && j+1==n)
                        {
                                max1 = output[i][j];
                        }
                        else if (i+1==m)
                        {
                                max1 = output[i][j]>=output[i][j+1]?output[i][j]:output[i][j+1];
                        }
                        else if (j+1==n)
                        {
                                max1 = output[i][j]>=output[i+1][j]?output[i][j]:output[i+1][j];
                        }
                        pooling_output[i/2][j/2] = max1;


                }

        }

		 auto end_time_pool = chrono::high_resolution_clock::now();
         cout<<"The time in microseconds for pooling layer is"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_pool - start_time_pool).count()<<" microseconds"<<endl;
         cout<<"The expected performance is 80Gp/s";
         long int pool_performance= ((m*n)/chrono::duration_cast<chrono::microseconds>(end_time_pool - start_time_pool).count());
         cout<<"The performance for pooling layer got is"<<pool_performance<<"Mpps"<<endl;

        /* cout<<"The rows and columns after pooling are"<<p_m<<" "<<p_n<<endl;

        for(i=0;i<p_m;i++)
        {
                for(j=0;j<p_n;j++)
                {
                        cout<<" "<<pooling_output[i][j];
                }
        cout<<endl;
        }*/
}


/****************************************************************************Fully Connected Layer********************************************************************************
 *
 * Function Name- Fully Connected Layer
 * Function Purpose- The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features                   for classifying the input image into various classes based on the training dataset
 *
 * *******************************************************************************************************************************************************************************/


void matrixMultiplication (float **feature_matrix, float **fullyconnected, float **multiplied_result, float **pooling_output, int p_m, int p_n, int key)
{

        int i,j;
        int total_pixels=p_m*p_n;
        printf("Total pixels are %d/n",total_pixels);

        for(i=0;i<p_m;i++)
        {
                for(j=0;j<p_n;j++)
                {
                        feature_matrix[0][key]= pooling_output[i][j];
                        key=key+1;
                }
        }


        for(i=0;i<p_m*p_n;i++)
        {
                for(j=0;j<100;j++)
                {
                        fullyconnected[i][j]=1.0;
                }

        }
        int k_i, sum = 0;

        auto start_time_mul = chrono::high_resolution_clock::now();

        for (j=0;j<100;j++)
        {
                for (k_i=0;k_i<p_m*p_n;k_i++)
                {

                        sum = sum + feature_matrix [0][k_i]*fullyconnected[k_i][j];
                }

                multiplied_result[0][j] = sum;
                sum = 0;
        }

 auto end_time_mul= chrono::high_resolution_clock::now();
        cout<<"The time in microseconds for fully connected layer is"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_mul - start_time_mul).count()<<" microseconds"<<endl;
        int fc_performance=((total_pixels*100)/ chrono::duration_cast<chrono::microseconds>(end_time_mul - start_time_mul).count());
        int mem_bound= ((100*total_pixels)/((1*total_pixels)+(total_pixels*100)+(1*100)));
        cout<<"The flop bound performance for matrix multiplication is expected to be 8.5Gp/s"<<endl;
        cout<<"The mem bound performance for matrix multiplication is expected to be"<<mem_bound<<"Gp/s"<<endl;
        cout<<"The actual perfromance got for matrix multiplication is"<<fc_performance<<"Mpps"<<endl;



        /*cout<<"Feature matrix is"<<endl;

        for (i=0;i<p_m*p_n;i++)
        {
                cout<<feature_matrix[0][i]<<" ";
        }

        cout<<endl<<"fully connected matrix is"<<endl;

        for (i=0;i<p_m*p_n;i++)
        {
                for (j=0;j<100;j++)
                {
                        cout<<fullyconnected[i][j]<<" ";
                }
                cout<<endl;
        }

        cout<<"Output after matrix multiplication is "<<endl;

        for(j=0;j<100;j++)
        {
                cout<<multiplied_result[0][j]<<" ";
        }*/

}




		
 