#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <chrono>
#include "lodepng.h"
#include <cstdlib>
using namespace std;

bool TESTMODE = true;

__global__ void maxPool(unsigned char* ip_img, unsigned char* op_img, unsigned int* d_width, unsigned int* d_pix_per_thread, unsigned int* d_op_pixels) {
	int startIndex = threadIdx.x * *d_pix_per_thread;
	int op_width = *d_width / 2;
	printf("%d\n", *d_pix_per_thread);
	int curr_index = startIndex;

	for (int i{ 0 }; i < *d_pix_per_thread; i++) {
		//convert into catesian coods
		int op_coods[2], ip_coods[2];
		op_coods[0] = curr_index % op_width;
		op_coods[1] = curr_index / op_width;

		ip_coods[0] = 2 * op_coods[0];
		ip_coods[1] = 2 * op_coods[1];
			
		//convert back from cartesian to linear
		int ip_idx = 4 * (ip_coods[0] + ip_coods[1] * *d_width);
		
		//search the 2x2 for the max value of each layer
		unsigned char max_rgba[4] = { 0,0,0,0 };
		for (int j{ 0 }; j < 4; j++) {
			int target = ip_idx;

			switch (j)	//switch to visit each pixel in the 2x2
			{
			case 0: break; //we already computed the starting pixel ip_idx
			case 1: target += 4; break;	//one pixel to the right
			case 2: target += 4 * *d_width; break;	//one row below
			case 3: target += 4 * *d_width + 4; break;	//down and right
			default: printf("This should be unreachable....."); break;
			}

			//check if we beat the old maximum on any layer
			for (int k{ 0 }; k < 4; k++) {
				if (ip_img[target + k] > max_rgba[k])
					max_rgba[k] = ip_img[target + k];
			}

		}
		
		//write to output
		for (int j{ 0 }; j < 4; j++)
			op_img[4 * curr_index + j] = max_rgba[j];			//TODO replace RHS
		curr_index++;
		if (curr_index > * d_op_pixels) return;			//overflow protection
	}
}

vector<unsigned char> singleThreadedPool(vector<unsigned char> ip_img, int ip_width, int ip_height) {
	printf("hit");
	vector<unsigned char> output;
	
	int op_width = ip_width / 2;
	int op_height = ip_height / 2;
	
	//loop for each output pixel (in order)
	for (int op_y{ 0 }; op_y < op_height; op_y++) {
		for (int op_x{ 0 }; op_x < op_width; op_x++) {
			//get the cartesion for the input
			int ip_coord[2];
			ip_coord[0] = 2 * op_x;
			ip_coord[1] = 2 * op_y;
			int ip_idx = 4 * (ip_coord[0] + ip_coord[1] * ip_width);

			//search the pixels in the 2x2
			unsigned char max_rgba[4] = { 0,0,0,0 };
			for (int i{ 0 }; i < 4; i++) {
				int target = ip_idx;

				switch (i) {	//switch to visit each pixel
				case 0: break;
				case 1: target += 4; break;
				case 2: target += 4 * ip_width; break;
				case 3: target += 4 * ip_width; break;
				default: printf("This should be unreachable...."); break;
				}

				//check to see if we have a new winner
				for (int j{ 0 }; j < 4; j++) {
					if (ip_img[target + j] > max_rgba[j])
						max_rgba[j] = ip_img[target + j];
				}
			}

			//write to output
			for (int i{ 0 }; i < 4; i++)
				output.push_back(max_rgba[i]);

		}
	}
	return output;

}

int main(int argc, char* argv[]) {
	//default CLAs
	string ip_img_name = "test.png";
	string op_img_name = "output.png";
	int num_threads = 1;

	//get command line args
	if (argc >= 2) ip_img_name = string(argv[1]);
	if (argc >= 3) op_img_name = string(argv[2]);
	if (argc >= 4) num_threads = atoi(argv[3]);

	vector<unsigned char> img;
	unsigned int width, height;

	//load the image
	unsigned error = lodepng::decode(img, width, height, ip_img_name);
	//if (error) cout << "Error loading image: " << error << ": " << lodepng_error_text(error) << endl;

	if (num_threads == 1) {
		auto start = chrono::high_resolution_clock::now();

		vector<unsigned char> output = singleThreadedPool(img, width, height);

		auto end = chrono::high_resolution_clock::now();
	
		error = lodepng::encode(op_img_name, output, width/2, height/2);

		auto time = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
		if (TESTMODE)
			cout << time << endl;
	}
	else{
		int op_width = width / 2;
		int op_height = height / 2;
		int op_pixels = op_width * op_height * 4;
		int op_size = op_pixels * sizeof(char);
		int ip_size = size(img) * sizeof(char);

		int pixels_per_thread = ((op_width * op_height) + 1) / num_threads;

		//begin timer for parallel overhead
		auto begin_overhead = chrono::high_resolution_clock::now();

		//device memory allocation
		unsigned char* d_ip_img;
		unsigned char* d_op_img;
		unsigned int* d_width;
		unsigned int* d_pix_per_thread;
		unsigned int* d_op_pixels;
		cudaMalloc((void**)& d_ip_img, ip_size);
		cudaMalloc((void**)& d_op_img, op_size);
		cudaMalloc((void**)& d_width, sizeof(int));
		cudaMalloc((void**)& d_pix_per_thread, sizeof(int));
		cudaMalloc((void**)& d_op_pixels, sizeof(int));

		//host memory allocation
		unsigned char* ip_img_ptr = (unsigned char*)malloc(ip_size);
		for (int i{ 0 }; i < size(img); i++)
			ip_img_ptr[i] = img[i];
		unsigned char* op_img_ptr = (unsigned char*)malloc(op_size);

		//copy to device
		cudaMemcpy(d_ip_img, ip_img_ptr, ip_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_pix_per_thread, &pixels_per_thread, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_op_pixels, &op_pixels, sizeof(int), cudaMemcpyHostToDevice);

		/******************** Parallel Computations ********************/
		auto being_parallel = chrono::high_resolution_clock::now();
	
		maxPool<<<1, num_threads >>> (d_ip_img, d_op_img, d_width, d_pix_per_thread, d_op_pixels);

		auto end_parallel = chrono::high_resolution_clock::now();
		/************************* End Parallel *************************/

		//copy data from device
		cudaMemcpy(op_img_ptr, d_op_img, op_size, cudaMemcpyDeviceToHost);

		//free device
		cudaFree(d_ip_img); cudaFree(d_op_img);
		cudaFree(d_width); cudaFree(d_pix_per_thread);

		auto end_overhead = chrono::high_resolution_clock::now();

		//convert back into a vector
		vector<unsigned char> op_vector;
		for (int i{ 0 }; i < (op_height * op_width) * 4; i++)
			op_vector.push_back(op_img_ptr[i]);

		error = lodepng::encode(op_img_name, op_vector, op_width, op_height);
		//if (error)
		//	cout << "Encode error: " << error << ": " << lodepng_error_text(error) << endl;

		//cleanup
		free(op_img_ptr); free(ip_img_ptr);

		//display timing data
		auto time_with_overhead = chrono::duration_cast<chrono::nanoseconds>(end_overhead - begin_overhead).count();
		auto time_parallel = chrono::duration_cast<chrono::nanoseconds>(end_parallel - begin_overhead).count();
		if (TESTMODE)
			cout << time_with_overhead << ',' << time_parallel << endl;
	}
	return 0;
}