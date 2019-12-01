#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <bits/stdc++.h> 
#include <chrono> 
using namespace std; 

int main(int argc, char** argv){

	if(argc!=2){
		printf("usage ./reduction_cpu size\n");
		return 1;
	}
	long long int size=atoi(argv[1]);
	printf("Array length is %lli int.\n",size);
	while(!(size%2)) size/=2;
	if(size!=1){
		printf("Array size is no power of two. Aborting\n");
		return 1;
	}
	size=atoi(argv[1]);
	//time_t init,fin;
	long long int* array = (long long int*) malloc(size*sizeof(long long int));
	for(int i=0;i<size;i++){
		array[i]=i+1;
	}
	long long int gauss=size*(size+1)/2;

	//init=clock();
	auto start = chrono::high_resolution_clock::now(); 
	for(int stride=2;stride<=size;stride*=2){	
		for(int j=0;j+stride/2<size;j+=stride){
			array[j]+=array[j+stride/2];
			//array[j+stride/2]=0;
		}
	 }
	//fin=clock();
	auto end = chrono::high_resolution_clock::now();
	double time_taken =  
      chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
    time_taken *= 1e-3; 
	printf("GSR %lli Gauss %lli Time in µs %f\n", array[0], gauss,time_taken);//(fin-init)*1e6/(CLOCKS_PER_SEC)
	return 0;
	}
	
		
