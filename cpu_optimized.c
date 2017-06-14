#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <math.h>
using namespace std;
void initialize (uint32_t m, uint32_t n, uint32_t k, int p_m, int p_n, float **R_image, float **G_image, float **B_image,
                                        float **R_kernel, float **G_kernel, float **B_kernel, float **output, float **pooling_output,
                                                                        float **feature_matrix, float **fullyconnected, float **multiplied_result);

void convolution (float **output, float **R_image, float **G_image, float **B_image, float **R_kernel, float **G_kernel, float **B_kernel,
        int total_rblocks, int total_cblocks, uint32_t r_block, uint32_t c_block, int kernelcenterX, int kernelcenterY, uint32_t m, uint32_t n, uint32_t k);

void rectifiedLinearUnit (float **output, int m, int n, int max);
void poolingLayer (float **output, float **pooling_output, int m, int n, int p_m, int p_n);
void matrixMultiplication (float **feature_matrix, float **fullyconnected, float **multiplied_result, float **pooling_output, int p_m, int p_n, int key);

int main(int argc, char** argv)
 {

    if (argc!= 6)
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

        uint32_t r_block= atoi(argv[4]);
        cout<<"R_block is "<<r_block << endl<<endl;

        uint32_t c_block= atoi(argv[5]);
        cout<<"C_block is "<<c_block << endl<<endl;

        int total_rblocks= ceil(m/r_block)+1;
        int total_cblocks=ceil(n/c_block)+1;
        printf("Total R_block are %d\n",total_rblocks);

        uint32_t k_center= (k*k)/2;
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
                                        R_kernel, G_kernel, B_kernel, output, pooling_output, feature_matrix, fullyconnected, multiplied_result);

        convolution (output, R_image, G_image, B_image, R_kernel, G_kernel, B_kernel, total_rblocks, total_cblocks, r_block,
                                c_block, kernelcenterX, kernelcenterY, m, n, k);


        rectifiedLinearUnit (output, m, n, max);

        poolingLayer (output, pooling_output, m, n, p_m, p_n);


        matrixMultiplication (feature_matrix, fullyconnected, multiplied_result, pooling_output, p_m, p_n, key);


}


void initialize (uint32_t m, uint32_t n, uint32_t k, int p_m, int p_n, float **R_image, float **G_image, float **B_image,
float **R_kernel, float **G_kernel, float **B_kernel, float **output, float **pooling_output, float **feature_matrix, float **fullyconnected, float **multiplied_result)

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


void convolution (float **output, float **R_image, float **G_image, float **B_image, float **R_kernel, float **G_kernel, float **B_kernel,
int total_rblocks, int total_cblocks, uint32_t r_block, uint32_t c_block, int kernelcenterX, int kernelcenterY, uint32_t m, uint32_t n, uint32_t k)
{

        int i, j;
                /* Start Convolution*/
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(16); // Use 4 threads for all consecutive parallel region

        auto start_time = chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic,512) collapse(2)

        for(int blockR=0;blockR<total_rblocks;blockR++)
        {
                for(int blockcol=0;blockcol<total_cblocks;blockcol++)
                {
                        for(int rows=blockR*r_block; rows<std::min((blockR+1)*r_block, m); ++rows)
                        {
                                for(int columns=blockcol*c_block; columns<std::min((blockcol+1)*c_block, n); ++columns)
                                {
                                        for(int krows=0;krows<k;krows++)
                                        {
                                                int mm = k - 1 - krows;

                                                for(int kcolumns=0; kcolumns<k;++kcolumns)
                                                {
                                                        int nn = k - 1 - kcolumns;
                                                        __m256  kern= _mm256_set1_ps(R_kernel[mm][nn]);
                                                        int ii = rows + (krows - kernelcenterY);
                                                        int jj = columns + (kcolumns -kernelcenterX);

                                                        if(ii>=0 && ii<m && jj>=0 && jj<n)
                                                        {

                                                                output[rows][columns] += (R_image[ii][jj] * R_kernel[mm][nn])+ (G_image[ii][jj] * G_kernel[mm][nn]) + ( B_image[ii][jj] * B_kernel[mm][nn]);
                                                        }


                                                }
                                        }

                                }
                        }
                }
        }


auto end_time = chrono::high_resolution_clock::now();
        int operation_performance= (819/(k*k));
        int mem_time= ((34*m*n)/ ((2*m*n)+ (k*k)));
        int total_pixels= (3*m*n);
        //out<<"size of pooling layer matrix"  <<p_m<<"  "<<p_n;
        //cout<<"Total_pixels are"<<total_pixels<<endl;

        cout<<"The performance for Convolution is as belows"<<endl;

        cout<<"The time in microseconds for convolution"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<" microseconds"<<endl;
        int actual= (total_pixels/(int) chrono::duration_cast<chrono::microseconds>(end_time - start_time).count());
        cout<<"The floptime performance for floating point operation is"<<operation_performance<<" Gigapixels"<<endl;
        cout<<"Performance for Memory Transfer "<<mem_time<<" Gigapixels/sec"<<endl;
        printf("The actual performance achieved is %d\n Megapixels/sec",actual);
        //cout<<"The matrix after convolution is"<<endl;
        /* for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        cout<<" "<<output[i][j];
                }
                cout<<endl;
        }

        */
}

void rectifiedLinearUnit (float **output, int m, int n, int max)
{

        int i, j;

        auto start_time_1 = chrono::high_resolution_clock::now();


        #pragma omp parallel for schedule(dynamic,512) collapse(2)

        for(i=0;i<m;i++)
        {
                for(j=0;j<n-n%8;j+=8)
                {
                        __m256 veca = _mm256_loadu_ps(&output[i][j]);
                        __m256 vecb = _mm256_set1_ps(max);
                        _mm256_storeu_ps (&output[i][j], vecb);

                }
        }

        auto end_time_1 = chrono::high_resolution_clock::now();

        cout<<endl<<"The time in microseconds for matrix avx"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_1 - start_time_1).count()<<" microseconds"<<endl;
        cout<<"The expected performance is Memory Bound and the value is 50Gp/s";
        int relu_performance=((m*n)/ chrono::duration_cast<chrono::microseconds>(end_time_1 - start_time_1).count());
        cout<<"The performance obtained was"<<relu_performance<<"Mp/s"<<endl;

        //cout<<"The final matrix after rectifiled linear unit is"<<endl;
         /*for(i=0;i<m;i++)
         {
                 for(j=0;j<n;j++)
                  {
                          cout<<" "<<output[i][j];
                  }
                  cout<<endl;
         }*/

}


void poolingLayer (float **output, float **pooling_output, int m, int n, int p_m, int p_n)
{

        int i, j, max1, max2;
        auto start_time_2 = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic,512) collapse(2)
        for(i=0; i< m-m%2; i+=2)

        {
                for(j=0;j< n;j+=8)
                {
                        __m256 veca = _mm256_loadu_ps(&output[i][j]);
                        __m256 vecb = _mm256_loadu_ps(&output[i+1][j]);
                        __m256 vecc = _mm256_max_ps(veca, vecb);
                        __m256 vecd = _mm256_set_ps(0, 0, 0, 0, vecc[6]>vecc[7]?vecc[6]:vecc[7],
                                                                        vecc[4]>vecc[5]?vecc[4]:vecc[5], vecc[2]>vecc[3]?vecc[2]:vecc[3],
                                                                                                                                        vecc[0]>vecc[1]?vecc[0]:vecc[1]);
                        _mm256_storeu_ps (&pooling_output[i/2][j/2], vecd);
                }
        }

        auto end_time_2 = chrono::high_resolution_clock::now();
        cout<<endl<<"The time in microseconds for pooling layer"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count()<<" microseconds"<<endl;
        cout<<"The expected performance is Memory Bound and the value is 80Gp/s";
        int pool_performance=((m*n)/ chrono::duration_cast<chrono::microseconds>(end_time_2 - start_time_2).count());
        cout<<"The performance obtained was"<<pool_performance<<"Mp/s"<<endl;



        cout<<"The rows and columns after pooling are"<<p_m<<" "<<p_n<<endl;




        /*for(i=0;i<p_m;i++)
         {
                for(j=0;j<p_n;j++)
                {
                         cout<<" "<<pooling_output[i][j];
                }
        cout<<endl;
        }*/

}

void matrixMultiplication (float **feature_matrix, float **fullyconnected, float **multiplied_result, float **pooling_output, int p_m, int p_n, int key)
{

        int i,j;

        int total_pixels = p_m*p_n;
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

        int k_i;
        float sum = 0, sum1[100];

        float scratchpad[8];

        auto start_time_3 = chrono::high_resolution_clock::now();
        //#pragma omp parallel for schedule(dynamic,1024)
        //sum=0;

 for (j=0;j<100;j++)
        {
                for (k_i=0;k_i<(p_m*p_n) - (p_m*p_n)%8;k_i+=8)
                {


                        __m256 veca = _mm256_loadu_ps(&feature_matrix[0][k_i]);
                        __m256 vecb = _mm256_loadu_ps(&fullyconnected[k_i][j]);
                        __m256 vecc = _mm256_mul_ps(veca, vecb);
                        for (i=0;i<8;i++)
                        {
                                sum+=vecc[i];
                        }

                }

                multiplied_result[0][j] = sum;
                sum = 0;

        }
        auto end_time_3 = chrono::high_resolution_clock::now();

        cout<<endl<<"The time in microseconds for matrix multiplication"<<std::fixed<< chrono::duration_cast<chrono::microseconds>(end_time_3 - start_time_3).count()<<" microseconds"<<endl;


        float fc_performance=((total_pixels)/(float) chrono::duration_cast<chrono::microseconds>(end_time_3 - start_time_3).count());
        float mem_bound= ((100*total_pixels)/(float)((1*total_pixels)+(total_pixels*100)+(1*100)));
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
        }*/


        cout<<"Output after matrix multiplication is "<<endl;

        /*for(j=0;j<100;j++)
        {
                cout<<multiplied_result[0][j]<<" ";

        }*/
}








		