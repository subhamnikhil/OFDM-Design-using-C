#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "sds_lib.h"

    #define  FFTSize 4 //size of FFT
    #define  pi 3.14159265
	#define cp 2
	#define iter 100


//channel desin AWGN
double rand_val(int seed)
{
  const long  a =      16807;  // Multiplier
  const long  m = 2147483647;  // Modulus
  const long  q =     127773;  // m div a
  const long  r =       2836;  // m mod a
  static long x;               // Random int value
  long        x_div_q;         // x divided by q
  long        x_mod_q;         // x modulo q
  long        x_new;           // New x value

  // Set the seed if argument is non-zero and then return zero
  if (seed > 0)
  {
    x = seed;
    return(0.0);
  }

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0)
    x = x_new;
  else
    x = x_new + m;

  // Return a random value between 0.0 and 1.0
  return((float) x / m);
}

//channel desin AWGN
double norm(float mean, float std_dev)
{
	float   u, r, theta;           // Variables for Box-Muller method
	float   x;                     // Normal(0, 1) rv
	float   norm_rv;               // The adjusted normal rv

  // Generate u
  u = 0.0;
  while (u == 0.0)
    u = rand_val(0);

  // Compute r
  r = sqrt(-2.0 * log(u));

  // Generate theta
  theta = 0.0;
  while (theta == 0.0)
    theta = 2.0 * pi * rand_val(0);

  // Generate x value
  x = r * cos(theta);

  // Adjust x value for specified mean and variance
  norm_rv = (x * std_dev) + mean;

  // Return the normally distributed RV value
  return(norm_rv);
}


//channel
void AWGN_Channel_real(float *noise_bpsk){

    //srand(time(0));
    rand_val(rand()+1);
    int mean=0;
    int std_dev =1;

  for (int i=0; i<FFTSize+cp; i++)
  {
    // Generate a normally distributed rv
    noise_bpsk[i] = norm(mean, std_dev)*(1/sqrt(2));

    // // Output the norm_rv value
    // fprintf(fp_out, "%f \n", norm_rv);
  //  printf("AWGN_Channel_Output:%lf ",noise_bpsk[i]);
  }

}

//channel
void AWGN_Channel_img(float *noise_bpsk_img,float *noise_bpsk){

    //srand(time(0));
    rand_val(rand()%1000+1);
    int mean=0;
    int std_dev =1;

  for (int i=0; i<FFTSize+cp; i++)
  {
    // Generate a normally distributed rv
    noise_bpsk_img[i] = norm(mean, std_dev)*(1/sqrt(2));

    // // Output the norm_rv value
    // fprintf(fp_out, "%f \n", norm_rv);
    printf("AWGN_Channel_Output:%f + j* %f \n",noise_bpsk[i],noise_bpsk_img[i]);
  }

}

void randNumber(float *data){
//srand(time(0)); // Use current time as seed for random generator
	for(int i = 0;i<FFTSize;i++)
{
        data[i]=rand()%2;
		printf("DataInput:%f ",data[i]);
}

}


void bpskNumberGenerator(float *data,float *x_k_bpsk){

	for(int i = 0;i<FFTSize;i++)
{
        x_k_bpsk[i]=(1/sqrt(2))*(2*data[i]-1);
		printf("BPSKInput:%f ",x_k_bpsk[i]);
}

}

//
void bpskNumberGeneratorRX(float *x_k_fft,float *dataout){

	for(int i = 0;i<FFTSize;i++)
{
        dataout[i]=(x_k_fft[i]+1)/2;
		printf("dataOut: %f ",dataout[i]);
}

}
float error=0;
void OverallBitErrorCalculation(float *data,float *dataout,float *OverallBitError,int k){
	for (int i = 0; i < FFTSize; i++) {
		OverallBitError[k]=OverallBitError[k]+abs((data[i]-dataout[i])/FFTSize);
    }
	error=error+OverallBitError[k];
	printf("BitError: %f for  Iteration %d",OverallBitError[k],k+1);
	printf("\n");
//	printf("OverallBitError: %f ",error);
//OverallBitError[k]=error;
}

void compareIPandOutput(float *data,float *dataout){
	int i;
	for (i = 0; i < FFTSize; i++) {
         if (data[i] != dataout[i]) {
             printf("Mismatch at data index %d where Actual output %f and Expected output %lf\n",i,data[i],dataout[i]);
				break;
         }
    }

	for(int j = 0;j<FFTSize;j++)
{
		printf("dataIN: %f ",data[j]);
}
	printf("\n");
	for(int j = 0;j<FFTSize;j++)
{
		printf("dataOut: %f ",dataout[j]);
}
	printf("\n");
if(i==FFTSize)
printf("PASS-GOODWORK");
else
printf("FAIL");
}

void initializeZeroes(float *x_n_ifft,float *x_n_ifft_img,float *x_k_fft,float *x_k_fft_img,float *OverallBitError){

	for(int i = 0;i<FFTSize;i++)
{
        x_n_ifft[i]=0.00;
		x_n_ifft_img[i]=0.00;
		x_k_fft[i]=0.00;
		x_k_fft_img[i]=0.00;
	//	printf("BPSKInput:%lf ",x_k_ifft[i]);
}

for (int i = 0; i < iter; i++)
{
  OverallBitError[i]=0;
}


}

void IFFTNumberGenerator(float *x_k_bpsk,float *x_n_ifft,float *x_n_ifft_img){

// initializeZeroes(x_n_ifft);
// initializeZeroes(x_n_ifft_img);

	for(int n = 0;n<FFTSize;n++)
{
    for(int k=0;k<FFTSize;k++)
    {
        //  x_k_ifft[n]=x_k_ifft[n]+x_k_bpsk[k]*exp((sqrt(-1)*2*pi*k*n)/FFTSize);
		x_n_ifft[n]=x_n_ifft[n]+x_k_bpsk[k]*cos((2*pi*k*n)/FFTSize);
		x_n_ifft_img[n]=x_n_ifft_img[n]+x_k_bpsk[k]*sin((2*pi*k*n)/FFTSize);
		//printf("IFFTInput:%lf ",x_k_ifft[n]);
    }
    x_n_ifft[n]=x_n_ifft[n]/FFTSize;
	x_n_ifft_img[n]=x_n_ifft_img[n]/FFTSize;
   printf("IFFTInput:%f j* %f \n",x_n_ifft[n],x_n_ifft_img[n]);
}



}


void FFTNumberGenerator(float *x_k_cyclic,float *x_k_cyclic_img,float *x_k_fft,float *x_k_fft_img){


	for(int n = 0;n<FFTSize;n++)
{
    for(int k=0;k<FFTSize;k++)
    {
        //  x_k_ifft[n]=x_k_ifft[n]+x_k_bpsk[k]*exp((sqrt(-1)*-2*pi*k*n)/FFTSize);
    	//Euler expression is exp(ix)=cosx+i*sinx and x(t)=Re(x)+i*Im(x)
    	//so finally this x(t)*exp((2*pi*i*t*k)/n)=[Re(x(t))+i*Im(x(t))]*[cos(2*pi*t*k/n) - i*sin(2*pi*t*k/n)]

		x_k_fft[n]=x_k_fft[n]+( x_k_cyclic[k]*cos((2*pi*k*n)/FFTSize) + x_k_cyclic_img[k]*sin((2*pi*k*n)/FFTSize)  );
		x_k_fft_img[n]= x_k_fft_img[n] + (   (-1)*x_k_cyclic[k]*sin((2*pi*k*n)/FFTSize) +  x_k_cyclic_img[k]*cos((2*pi*k*n)/FFTSize)  );

    }

   printf("FFTOutput:%f +j*%f \n",x_k_fft[n],x_k_fft_img[n]);
}


}

void CyclicPrefixTxGenerator(float *x_n_ifft,float *x_n_ifft_img,float *x_n_cyclic,float *x_n_cyclic_img){

	for(int i = FFTSize+cp-1;i>=0;i--)
{
        if(i<cp){
		x_n_cyclic[i]=x_n_ifft[FFTSize-1-i];
		x_n_cyclic_img[i]=x_n_ifft_img[FFTSize-1-i];
		}
		else{
		x_n_cyclic[i]=x_n_ifft[i-cp];
		x_n_cyclic_img[i]=x_n_ifft_img[i-cp];

		}
		printf("cyclicTx:%f j* %f \n",x_n_cyclic[i],x_n_cyclic_img[i]);
}

}


void CyclicPrefixRxGenerator(float *rec_bpsk,float *rec_bpsk_img,float *x_k_cyclic,float *x_k_cyclic_img){

	for(int i = 0;i<FFTSize;i++)
{
        x_k_cyclic[i]=rec_bpsk[i+cp];
		x_k_cyclic_img[i]=rec_bpsk_img[i+cp];
		printf("cyclicRx:%f j* %f \n",x_k_cyclic[i],x_k_cyclic_img[i]);
}

}

void dicisionRule(int M,float *x_k_fft,float *dicisionout){
//=[1 2] ,=[-1.00 1.00]

double m[M];
double M_sym_set[M],t[FFTSize];
for(int i=0;i<M;i++)
{
m[i]=i+1;
M_sym_set[i]=2*m[i] - 1 - M;
}

for(int i=0;i<FFTSize;i++)
{
	for(int k=0;k<M;k++)
	{
    t[k] = (x_k_fft[i]-M_sym_set[k])*(x_k_fft[i]-M_sym_set[k]);
	}

		if(t[0]<t[M-1])
		{
			dicisionout[i]=M_sym_set[0];
		}
		else
		{
			dicisionout[i]=M_sym_set[M-1];
		}
printf("DicisionOut:%f\n",dicisionout[i]);
}


}

void AWGN_Channel_add_with_CP(float *noise_bpsk,float *noise_bpsk_img,float *x_n_cyclic,float *x_n_cyclic_img,float *rec_bpsk,float *rec_bpsk_img){

		for(int i = 0;i<FFTSize+cp;i++)
{
        rec_bpsk[i]=x_n_cyclic[i]+noise_bpsk[i];
		rec_bpsk_img[i]=x_n_cyclic_img[i]+noise_bpsk_img[i];
		printf("AWGN_Channel_add_with_CP:%f j* %f \n",rec_bpsk[i],rec_bpsk_img[i]);
}
}


int  main()
{

	float *x_k_bpsk,*x_k_fft,*rec_bpsk,*rec_bpsk_img,*OverallBitError;
    float *data,*dataout,*x_k_fft_img,*x_n_ifft_img,*x_n_ifft,*x_n_cyclic,*x_n_cyclic_img,*x_k_cyclic,*x_k_cyclic_img,*dicisionout,*noise_bpsk,*noise_bpsk_img;
	// double complex *test1;

	//allocate space for matrices in memory
	x_k_fft = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_k_fft_img = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_n_ifft = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_n_ifft_img = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_n_cyclic = (float *)sds_alloc((FFTSize+cp)  * sizeof(float));
	x_n_cyclic_img = (float *)sds_alloc((FFTSize+cp)  * sizeof(float));
	x_k_cyclic = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_k_cyclic_img = (float *)sds_alloc(FFTSize  * sizeof(float));
	x_k_bpsk = (float  *)sds_alloc(FFTSize * sizeof(float));
	noise_bpsk = (float  *)sds_alloc((FFTSize+cp) * sizeof(float));
	noise_bpsk_img = (float  *)sds_alloc((FFTSize+cp) * sizeof(float));
	data = (float *)sds_alloc(FFTSize * sizeof(float));
	dataout = (float *)sds_alloc(FFTSize * sizeof(float));
	dicisionout = (float *)sds_alloc(FFTSize * sizeof(float));
	rec_bpsk = (float *)sds_alloc((FFTSize+cp) * sizeof(float));
	rec_bpsk_img = (float *)sds_alloc((FFTSize+cp) * sizeof(float));
	OverallBitError = (float *)sds_alloc(iter * sizeof(float));
// data transfer symbol = iter*number of bit(FFT size)
for(int k=0;k<iter;k++){
//Tx
printf("\n################################# Start of %d iteration ####################################################\n",k+1);
printf("\n--------- Tx side Generation of random data in binary form------------------\n");
randNumber(data);
printf("\n");
printf("\n--------- Tx side Generation of BPSK signal with normalization factor-------------------\n");
bpskNumberGenerator(data,x_k_bpsk);
printf("\n");
printf("\n");
initializeZeroes(x_n_ifft,x_n_ifft_img,x_k_fft,x_k_fft_img,OverallBitError);
printf("\n--------- Tx side Performing IDFT on BPSK signal -------------------\n");
IFFTNumberGenerator(x_k_bpsk,x_n_ifft,x_n_ifft_img);
printf("\n");
printf("\n--------- Tx side Adding cyclic prefix to IDFT signal -------------------\n");
CyclicPrefixTxGenerator(x_n_ifft,x_n_ifft_img,x_n_cyclic,x_n_cyclic_img);
printf("\n");
//channel AWGN Noise
 AWGN_Channel_real(noise_bpsk);
 printf("\n");
 AWGN_Channel_img(noise_bpsk_img,noise_bpsk);
 printf("\n");
 printf("\n--------- AWGN noise addition in Tx  -------------------\n");
 AWGN_Channel_add_with_CP(noise_bpsk,noise_bpsk_img,x_n_cyclic,x_n_cyclic_img,rec_bpsk,rec_bpsk_img);
//Rx
printf("\n");

printf("\n--------- Rx side Removing cyclic prefix from channel signal -------------------\n");
CyclicPrefixRxGenerator(rec_bpsk,rec_bpsk_img,x_k_cyclic,x_k_cyclic_img);
printf("\n");
printf("\n--------- Rx side Performing DFT on Rx_side_Cyclic_data -------------------\n");
FFTNumberGenerator(x_k_cyclic,x_k_cyclic_img,x_k_fft,x_k_fft_img);
printf("\n");
printf("\n--------- Rx side ML decoding of DFT signal -------------------\n");
dicisionRule(2,x_k_fft,dicisionout);
printf("\n--------- Rx side Demodulation and generation of BPSK signal -------------------\n");
printf("\n");
bpskNumberGeneratorRX(dicisionout,dataout);
printf("\n");
printf("\n--------- Calculation of Bit Error probability-------------------\n");
OverallBitErrorCalculation(data,dataout,OverallBitError,k);
printf("\n");
printf("\n--------- Comparison between Input data and Output data-------------------\n");
compareIPandOutput(data,dataout);
printf("\n");
printf("\n########################################################################################\n");
printf("\n########################################################################################\n");

}

printf("\n########################################################################################\n");
printf("Overall Bit Error for OFDM Model is :%f",error/iter);
printf("\n########################################################################################\n");




sds_free(x_k_fft);
sds_free(rec_bpsk_img);
sds_free(x_k_bpsk);
sds_free(noise_bpsk);
sds_free(noise_bpsk_img);
sds_free(data);
sds_free(dataout);
sds_free(x_k_fft);
sds_free(x_k_fft_img);
sds_free(x_n_ifft_img);
sds_free(x_n_ifft_img);
sds_free(x_n_cyclic);
sds_free(x_n_cyclic_img);
sds_free(x_k_cyclic);
sds_free(x_k_cyclic_img);
sds_free(dicisionout);
sds_free(OverallBitError);

return 0;
}
