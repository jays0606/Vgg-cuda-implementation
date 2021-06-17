#include "vgg16_cuda.h"
#define TILE_WIDTH 16
#define TILE_WIDTH_2 16
#define TILE_WIDTH_HALF 8
#define TILE_WIDTH_QUARTER 4
#define TILE_WIDTH_EIGHTH 2
#define FLT_MIN -3.40282e+38
#define FLT_EPSILON 1.19209e-07
using namespace std;
#include <cmath>

__global__ void normalize(const uint8_t* const image, float* input, int C, int H, int W, int W_grid, int tile_size) {

    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // input height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // input width 

    // Initialize variables
    float max_int = 255.0L;
    float mean = 0.5L;
    float var = 0.5L;

    // Normalize
    int base = b * (C * H * W) + c * (H * W) + h * (W) + w;
    input[base] = (image[base]/max_int - mean) / var;
}

__global__ void pad(float* input, float* output, int C, int H, int W, int P, int W_grid, int tile_size) {

  int b = blockIdx.x; // mini_batch
  int c = blockIdx.y; // input channel
  int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // input height
  int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // input width 

  int H_OUT = H+2*P;
  int W_OUT = W+2*P;

  int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
  int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h+P) * W_OUT + w+P;

  output[output_index] = input[input_base];
}

__global__ void conv_shared(float* input, float* output, float* weight, float* bias, int H, int W, int IC, int OC, int K, int W_grid, int tile_size)
{
    int H_OUT = H - (K - 1); 
    int W_OUT = W - (K - 1); 
    int X_TILE_WIDTH = tile_size + K-1;

    extern __shared__ float shmem[]; // size H_OUT * W_OUT + K*K
    float *X_shared = &shmem[0];
    float *W_shared = &shmem[X_TILE_WIDTH * X_TILE_WIDTH];
    // float *W_shared = &shmem[0];

    int b = blockIdx.x; // mini_batch
    int oc = blockIdx.y; // output channel
    int h0 = threadIdx.y; // row, width for tile for one thread
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z/W_grid) * tile_size; // row, width for block dim 
    int w_base = (blockIdx.z%W_grid) * tile_size;
    int h = h_base + h0;
    int w = w_base + w0;
    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;

    float val = 0;
    for (int ic=0; ic<IC; ic++){

        if ((h0 < K) && (w0 < K)) // Load Kernel weight to SM
            W_shared[h0*K + w0] = weight[oc * (IC*K*K) + ic * (K*K) + h0*K + w0]; 
        __syncthreads();

        for(int i=h; i<h_base + X_TILE_WIDTH; i+= tile_size) // Load input weight to SM
            for (int j=w; j<w_base + X_TILE_WIDTH; j += tile_size)
                X_shared[(i-h_base)*X_TILE_WIDTH + j-w_base] = input[b * (IC * H * W) + ic * (H * W) + i * (W) + j];
        __syncthreads();

        val += X_shared[(h0)*X_TILE_WIDTH + w0] * W_shared[0];
        val += X_shared[(h0)*X_TILE_WIDTH + w0+1] * W_shared[1];
        val += X_shared[(h0)*X_TILE_WIDTH + w0+2] * W_shared[2];

        val += X_shared[(h0+1)*X_TILE_WIDTH + w0] * W_shared[1*K];
        val += X_shared[(h0+1)*X_TILE_WIDTH + w0+1] * W_shared[1*K + 1];
        val += X_shared[(h0+1)*X_TILE_WIDTH + w0+2] * W_shared[1*K + 2];

        val += X_shared[(h0+2)*X_TILE_WIDTH + w0] * W_shared[2*K];
        val += X_shared[(h0+2)*X_TILE_WIDTH + w0+1] * W_shared[2*K + 1];
        val += X_shared[(h0+2)*X_TILE_WIDTH + w0+2] * W_shared[2*K + 2];
        __syncthreads();

        // for (int kh = 0; kh < K; kh++) // Calculate
        //     for (int kw = 0; kw < K; kw++) 
        //         val += X_shared[(h0+kh)*X_TILE_WIDTH + w0+kw] * W_shared[kh*K + kw];
        // __syncthreads();
        
        // int kernel_base = oc * (IC * K * K) + ic * (K * K);
        // val += X_shared[(h0)*X_TILE_WIDTH + w0] * weight[kernel_base + 0];
        // val += X_shared[(h0)*X_TILE_WIDTH + w0+1] * weight[kernel_base + 1];
        // val += X_shared[(h0)*X_TILE_WIDTH + w0+2] * weight[kernel_base + 2];

        // val += X_shared[(h0+1)*X_TILE_WIDTH + w0] * weight[kernel_base + 1*K];
        // val += X_shared[(h0+1)*X_TILE_WIDTH + w0+1] * weight[kernel_base + 1*K + 1];
        // val += X_shared[(h0+1)*X_TILE_WIDTH + w0+2] * weight[kernel_base + 1*K + 2];

        // val += X_shared[(h0+2)*X_TILE_WIDTH + w0] * weight[kernel_base + 2*K];
        // val += X_shared[(h0+2)*X_TILE_WIDTH + w0+1] * weight[kernel_base + 2*K + 1];
        // val += X_shared[(h0+2)*X_TILE_WIDTH + w0+2] * weight[kernel_base + 2*K + 2];
        // __syncthreads();

        // int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
        // val += input[input_base + 0] * W_shared[0];
        // val += input[input_base + 1] * W_shared[1];
        // val += input[input_base + 2] * W_shared[2];

        // val += input[input_base + 1*W + 0] * W_shared[1*K];
        // val += input[input_base + 1*W + 1] * W_shared[1*K + 1];
        // val += input[input_base + 1*W + 2] * W_shared[1*K + 2];

        // val += input[input_base + 2*W + 0] * W_shared[2*K];
        // val += input[input_base + 2*W + 1] * W_shared[2*K + 1];
        // val += input[input_base + 2*W + 2] * W_shared[2*K + 2];
        // __syncthreads();
        }
    output[output_index] = bias[oc] + val;    
}

__global__ void conv(float* input, float* output, float* weight, float* bias, int H, int W, int IC, int OC, int K, int W_grid, int tile_size)
{
    int b = blockIdx.x; // mini_batch
    int oc = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // output width

    int H_OUT = H - (K - 1); 
    int W_OUT = W - (K - 1); 
    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;

    float val = 0;
    for (int ic=0; ic<IC; ic++){
        int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
        int kernel_base = oc * (IC * K * K) + ic * (K * K);

        val += input[input_base + 0] * weight[kernel_base + 0];
        val += input[input_base + 1] * weight[kernel_base + 1];
        val += input[input_base + 2] * weight[kernel_base + 2];

        val += input[input_base + 1*W + 0] * weight[kernel_base + 1*K];
        val += input[input_base + 1*W + 1] * weight[kernel_base + 1*K + 1];
        val += input[input_base + 1*W + 2] * weight[kernel_base + 1*K + 2];

        val += input[input_base + 2*W + 0] * weight[kernel_base + 2*K];
        val += input[input_base + 2*W + 1] * weight[kernel_base + 2*K + 1];
        val += input[input_base + 2*W + 2] * weight[kernel_base + 2*K + 2];

        // for (int kh = 0; kh < K; kh++)
        //     for (int kw = 0; kw < K; kw++) 
        //         val += input[input_base + kh * (W) + kw] * weight[kernel_base + kh * (K) + kw];

    }
    output[output_index] = bias[oc] + val;
}

__global__ void relu(float* input, int C, int H, int W, int W_grid, int tile_size)
{
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // input height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // input width 

    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
    input[input_base] = max(input[input_base], (float)(0.0));
}

__global__ void pool(float* input, float* output, int C, int H, int W, int W_grid, int tile_size)
{   
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // output width 

    int scale = 2;
    int H_IN = H*scale;
    int W_IN = W*scale;
    float max_val = FLT_MIN;
    int input_base = b * (C * H_IN * W_IN) + c * (H_IN * W_IN) + h*scale * (W_IN) + w*scale;

    // Find maximum
    for (int sh = 0; sh < scale; sh++)
        for (int sw = 0; sw < scale; sw++) {
            float val = input[input_base + sh * (W_IN) + sw];
            if (val - max_val > FLT_EPSILON) max_val = val;
        }

    int output_index = b * (C * H * W) + c * (H * W) + h * (W) + w;
    output[output_index] = max_val;
}

__global__ void fc(float* input, float* output, float* weight, float* bias, int IC, int OC)
{
    int b = blockIdx.x; // mini batch
    int c = blockIdx.y; // output channel index

    float val = 0;
    for (int ic=0; ic<IC; ic++)
        val += weight[c * IC + ic] * input[b * IC + ic];

    output[b * OC + c] = bias[c] + val;
}

// float *c_C1_1_feature_map = new float[batch * conv1_1_out_channel * C1_1_size * C1_1_size];
// cudaMemcpy(c_C1_1_feature_map, d_C1_1_feature_map, sizeof(float) * batch * conv1_1_out_channel * C1_1_size * C1_1_size, cudaMemcpyDeviceToHost);
void print(int channel, int width, float* feature_map){
    cout << "GPU " << endl;
    int base = 0;
    for (int c=0; c<channel; c+=16){
        base += channel*width*width;
        for (int i=0; i<width/2; i++){
            for (int j=0; j<width/2; j++)
                cout << ceil(feature_map[base + i*width + j]*10)/10.0 << " ";
            cout << endl;
        } cout << endl;
    }
    cout << endl;
}
void vgg16_cuda::predict(int batch) {

    dim3 dimGrid_0(batch, input_channel, ceil((input_size*input_size)/(TILE_WIDTH*TILE_WIDTH))); 
    dim3 dimBlock_0(TILE_WIDTH, TILE_WIDTH, 1); 
    normalize <<< dimGrid_0, dimBlock_0 >>> (d_image, d_input, input_channel, input_size, input_size, ceil(input_size/TILE_WIDTH), TILE_WIDTH);
    
    //////////BLOCK 1/////////////////////////////////
    pad <<< dimGrid_0, dimBlock_0 >>> (d_input, d_input_padded, input_channel, input_size, input_size, conv1_1_padding_size, ceil(input_size/TILE_WIDTH), TILE_WIDTH);
        
    dim3 dimGrid_1_1(batch, C1_1_channel, ceil((C1_1_size*C1_1_size)/(TILE_WIDTH*TILE_WIDTH))); // batch*64*4
    dim3 dimBlock_1_1(TILE_WIDTH, TILE_WIDTH, 1); 
    // size_t shmem_size_1 = sizeof(float) * ((TILE_WIDTH + conv1_1_kernel_size-1)*(TILE_WIDTH+conv1_1_kernel_size-1) + conv1_1_kernel_size*conv1_1_kernel_size); // (H-K+1)*(H-K+1) + K*K
    // conv_shared <<< dimGrid_1_1, dimBlock_1_1, shmem_size_1 >>> (d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, input_size+2*conv1_1_padding_size,
    //     input_size+2*conv1_1_padding_size, conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size, ceil(C1_1_size/TILE_WIDTH), TILE_WIDTH);
    conv <<< dimGrid_1_1, dimBlock_1_1 >>> (d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, input_size+2*conv1_1_padding_size,
                                    input_size+2*conv1_1_padding_size, conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size, ceil(C1_1_size/TILE_WIDTH), TILE_WIDTH);
    relu <<< dimGrid_1_1, dimBlock_1_1 >>> (d_C1_1_feature_map, C1_1_channel, C1_1_size, C1_1_size, ceil(C1_1_size/TILE_WIDTH), TILE_WIDTH);
    pad <<< dimGrid_1_1, dimBlock_1_1 >>> (d_C1_1_feature_map, d_C1_1_feature_map_padded, C1_1_channel, C1_1_size, C1_1_size, conv1_2_padding_size, ceil(C1_1_size/TILE_WIDTH), TILE_WIDTH);

    dim3 dimGrid_1_2(batch, C1_2_channel, ceil((C1_2_size*C1_2_size)/(TILE_WIDTH*TILE_WIDTH)));  // batch*64*4
    dim3 dimBlock_1_2(TILE_WIDTH, TILE_WIDTH, 1); 
    // conv_shared <<< dimGrid_1_2, dimBlock_1_2, shmem_size_1 >>> (d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, C1_1_size+2*conv1_2_padding_size,
    //     C1_1_size+2*conv1_2_padding_size, conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size, ceil(C1_2_size/TILE_WIDTH), TILE_WIDTH);
    conv <<< dimGrid_1_2, dimBlock_1_2>>> (d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, C1_1_size+2*conv1_2_padding_size,
                                    C1_1_size+2*conv1_2_padding_size, conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size, ceil(C1_2_size/TILE_WIDTH), TILE_WIDTH);
    relu <<< dimGrid_1_2, dimBlock_1_2 >>> (d_C1_2_feature_map, C1_2_channel, C1_2_size, C1_2_size, C1_2_size/TILE_WIDTH, TILE_WIDTH);
    
    dim3 dimGrid_1_3(batch, S1_channel, ceil((S1_size*S1_size)/(TILE_WIDTH*TILE_WIDTH))); // batch*64*1
    dim3 dimBlock_1_3(TILE_WIDTH, TILE_WIDTH, 1); 
    pool <<< dimGrid_1_3, dimBlock_1_3 >>> (d_C1_2_feature_map, d_S1_feature_map, S1_channel, S1_size, S1_size, ceil(S1_size/TILE_WIDTH), TILE_WIDTH);
    
    // //////////BLOCK 2/////////////////////////////////
        
    dim3 dimGrid2_0(batch, S1_channel, ceil((S1_size*S1_size)/(TILE_WIDTH_2*TILE_WIDTH_2))); // batch*64*1   
    dim3 dimBlock2_0(TILE_WIDTH_2, TILE_WIDTH_2, 1); 
    pad <<< dimGrid2_0, dimBlock2_0 >>> (d_S1_feature_map, d_S1_feature_map_padded, S1_channel, S1_size, S1_size, conv2_1_padding_size, ceil(S1_size/TILE_WIDTH_2), TILE_WIDTH_2);

    dim3 dimGrid_2_1(batch, C2_1_channel, ceil((C2_1_size*C2_1_size)/(TILE_WIDTH_2*TILE_WIDTH_2))); // batch*128*1
    dim3 dimBlock_2_1(TILE_WIDTH_2, TILE_WIDTH_2, 1); 
    // size_t shmem_size_2 = sizeof(float) * ((TILE_WIDTH_2 + conv2_1_kernel_size-1)*(TILE_WIDTH_2+conv2_1_kernel_size-1) + conv2_1_kernel_size*conv2_1_kernel_size);
    // conv_shared <<< dimGrid_2_1, dimBlock_2_1, shmem_size_2 >>> (d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, S1_size+2*conv2_1_padding_size,
    //     S1_size+2*conv2_1_padding_size, conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size, ceil(C2_1_size/TILE_WIDTH_2), TILE_WIDTH_2);
    conv <<< dimGrid_2_1, dimBlock_2_1 >>> (d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, S1_size+2*conv2_1_padding_size,
                                    S1_size+2*conv2_1_padding_size, conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size, ceil(C2_1_size/TILE_WIDTH_2), TILE_WIDTH_2);
    relu <<< dimGrid_2_1, dimBlock_2_1 >>> (d_C2_1_feature_map, C2_1_channel, C2_1_size, C2_1_size, C2_1_size/TILE_WIDTH_2, TILE_WIDTH_2);    
    pad <<< dimGrid_2_1, dimBlock_2_1 >>> (d_C2_1_feature_map, d_C2_1_feature_map_padded, C2_1_channel, C2_1_size, C2_1_size, conv2_2_padding_size, ceil(C2_1_size/TILE_WIDTH_2), TILE_WIDTH_2);
    
    dim3 dimGrid_2_2(batch, C2_2_channel, ceil((C2_2_size*C2_2_size)/(TILE_WIDTH_2*TILE_WIDTH_2))); // batch*128*1
    dim3 dimBlock_2_2(TILE_WIDTH_2, TILE_WIDTH_2, 1); 
    // conv_shared <<< dimGrid_2_2, dimBlock_2_2, shmem_size_2>>> (d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, C2_1_size+2*conv2_2_padding_size,
    //     C2_1_size+2*conv2_2_padding_size, conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size, ceil(C2_2_size/TILE_WIDTH_2), TILE_WIDTH_2);
    conv <<< dimGrid_2_2, dimBlock_2_2 >>> (d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, C2_1_size+2*conv2_2_padding_size,
                                    C2_1_size+2*conv2_2_padding_size, conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size, ceil(C2_2_size/TILE_WIDTH_2), TILE_WIDTH_2);
    relu <<< dimGrid_2_2, dimBlock_2_2 >>> (d_C2_2_feature_map, C2_2_channel, C2_2_size, C2_2_size, ceil(C2_2_size/TILE_WIDTH_2), TILE_WIDTH_2);
        
    dim3 dimGrid_2_3(batch, S2_channel, ceil((S2_size*S2_size)/(TILE_WIDTH_HALF*TILE_WIDTH_HALF))); // batch*128*1
    dim3 dimBlock_2_3(TILE_WIDTH_HALF, TILE_WIDTH_HALF, 1); // 8*8*1
    pool <<< dimGrid_2_3, dimBlock_2_3 >>> (d_C2_2_feature_map, d_S2_feature_map, S2_channel, S2_size, S2_size, ceil(S2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);

    // //////////BLOCK 3/////////////////////////////////

    dim3 dimGrid3_0(batch, S2_channel, ceil((S2_size*S2_size)/(TILE_WIDTH_HALF*TILE_WIDTH_HALF))); // batch*128*1
    dim3 dimBlock3_0(TILE_WIDTH_HALF,TILE_WIDTH_HALF, 1); // 8*8*1
    pad <<< dimGrid3_0, dimBlock3_0 >>> (d_S2_feature_map, d_S2_feature_map_padded, S2_channel, S2_size, S2_size, conv3_1_padding_size, ceil(S2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    
    dim3 dimGrid3_1(batch, C3_1_channel, ceil((C3_1_size*C3_1_size)/(TILE_WIDTH_HALF*TILE_WIDTH_HALF))); // batch*256*1
    dim3 dimBlock3_1(TILE_WIDTH_HALF,TILE_WIDTH_HALF, 1);
    // size_t shmem_size_3 = sizeof(float) * ((TILE_WIDTH_HALF + conv3_1_kernel_size-1)*(TILE_WIDTH_HALF+conv3_1_kernel_size-1) + conv3_1_kernel_size*conv3_1_kernel_size);
    // conv_shared <<< dimGrid3_1, dimBlock3_1, shmem_size_3 >>> (d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, S2_size+2*conv3_1_padding_size,
    //     S2_size+2*conv3_1_padding_size, conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size, ceil(C3_1_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    conv <<< dimGrid3_1, dimBlock3_1 >>> (d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, S2_size+2*conv3_1_padding_size,
                                    S2_size+2*conv3_1_padding_size, conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size, ceil(C3_1_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    relu <<< dimGrid3_1, dimBlock3_1 >>> (d_C3_1_feature_map, C3_1_channel, C3_1_size, C3_1_size, C3_1_size/TILE_WIDTH_HALF, TILE_WIDTH_HALF);
    pad <<< dimGrid3_1, dimBlock3_1 >>> (d_C3_1_feature_map, d_C3_1_feature_map_padded, C3_1_channel, C3_1_size, C3_1_size, conv3_2_padding_size, ceil(C3_1_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);

    dim3 dimGrid3_2(batch, C3_2_channel, ceil((C3_2_size*C3_2_size)/(TILE_WIDTH_HALF*TILE_WIDTH_HALF))); // batch*256*1
    dim3 dimBlock3_2(TILE_WIDTH_HALF,TILE_WIDTH_HALF, 1);
    // conv_shared <<< dimGrid3_2, dimBlock3_2, shmem_size_3 >>> (d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, C3_1_size+2*conv3_2_padding_size,
    //     C3_1_size+2*conv3_2_padding_size, conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size, ceil(C3_2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    conv <<< dimGrid3_2, dimBlock3_2 >>> (d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, C3_1_size+2*conv3_2_padding_size,
                                    C3_1_size+2*conv3_2_padding_size, conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size, ceil(C3_2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    relu <<< dimGrid3_2, dimBlock3_2 >>> (d_C3_2_feature_map, C3_2_channel, C3_2_size, C3_2_size, ceil(C3_2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    pad <<< dimGrid3_2, dimBlock3_2 >>> (d_C3_2_feature_map, d_C3_2_feature_map_padded, C3_2_channel, C3_2_size, C3_2_size, conv3_3_padding_size, ceil(C3_2_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);

    dim3 dimGrid3_3(batch, C3_3_channel, ceil((C3_3_size*C3_3_size)/(TILE_WIDTH_HALF*TILE_WIDTH_HALF))); // batch*256*1
    dim3 dimBlock3_3(TILE_WIDTH_HALF,TILE_WIDTH_HALF, 1);
    // conv_shared <<< dimGrid3_3, dimBlock3_3, shmem_size_3 >>> (d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, C3_2_size+2*conv3_3_padding_size,
    //     C3_2_size+2*conv3_3_padding_size, conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size, ceil(C3_3_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    conv <<< dimGrid3_3, dimBlock3_3 >>> (d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, C3_2_size+2*conv3_3_padding_size,
                                    C3_2_size+2*conv3_3_padding_size, conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size, ceil(C3_3_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);
    relu <<< dimGrid3_3, dimBlock3_3 >>> (d_C3_3_feature_map, C3_3_channel, C3_3_size, C3_3_size, ceil(C3_3_size/TILE_WIDTH_HALF), TILE_WIDTH_HALF);

    dim3 dimGrid3_4(batch, S3_channel, ceil((S3_size*S3_size)/(TILE_WIDTH_QUARTER*TILE_WIDTH_QUARTER))); // batch*256*1
    dim3 dimBlock3_4(TILE_WIDTH_QUARTER,TILE_WIDTH_QUARTER, 1); // 4*4*1
    pool <<< dimGrid3_4, dimBlock3_4 >>> (d_C3_3_feature_map, d_S3_feature_map, S3_channel, S3_size, S3_size, ceil(S3_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    
    // //////////BLOCK 4/////////////////////////////////

    dim3 dimGrid4_0(batch, S3_channel, ceil((S3_size*S3_size)/(TILE_WIDTH_QUARTER*TILE_WIDTH_QUARTER))); // batch*256*1
    dim3 dimBlock4_0(TILE_WIDTH_QUARTER,TILE_WIDTH_QUARTER, 1); // 4*4*1
    size_t shmem_size_4 = sizeof(float) * ((TILE_WIDTH_QUARTER + conv4_1_kernel_size-1)*(TILE_WIDTH_QUARTER+conv4_1_kernel_size-1) + conv4_1_kernel_size*conv4_1_kernel_size);
    pad <<< dimGrid4_0, dimBlock4_0 >>> (d_S3_feature_map, d_S3_feature_map_padded, S3_channel, S3_size, S3_size, conv4_1_padding_size, ceil(S3_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
             
    dim3 dimGrid4_1(batch, C4_1_channel, ceil((C4_1_size*C4_1_size)/(TILE_WIDTH_QUARTER*TILE_WIDTH_QUARTER))); // batch*256*1
    dim3 dimBlock4_1(TILE_WIDTH_QUARTER,TILE_WIDTH_QUARTER, 1); // 4*4*1
    // conv_shared <<< dimGrid4_1, dimBlock4_1, shmem_size_4 >>> (d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, S3_size+2*conv4_1_padding_size,
    //     S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size, ceil(C4_1_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    conv <<< dimGrid4_1, dimBlock4_1 >>> (d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, S3_size+2*conv4_1_padding_size,
                                    S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size, ceil(C4_1_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    relu <<< dimGrid4_1, dimBlock4_1 >>> (d_C4_1_feature_map, C4_1_channel, C4_1_size, C4_1_size, ceil(C4_1_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    pad <<< dimGrid4_1, dimBlock4_1 >>> (d_C4_1_feature_map, d_C4_1_feature_map_padded, C4_1_channel, C4_1_size, C4_1_size, conv4_2_padding_size, ceil(C4_1_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    
    dim3 dimGrid4_2(batch, C4_2_channel, ceil((C4_2_size*C4_2_size)/(TILE_WIDTH_QUARTER*TILE_WIDTH_QUARTER))); // batch*256*1
    dim3 dimBlock4_2(TILE_WIDTH_QUARTER,TILE_WIDTH_QUARTER, 1); // 4*4*1
    // conv_shared <<< dimGrid4_2, dimBlock4_2,  shmem_size_4>>> (d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, C4_1_size+2*conv4_2_padding_size,
    //     C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size, ceil(C4_2_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    conv <<< dimGrid4_2, dimBlock4_2 >>> (d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, C4_1_size+2*conv4_2_padding_size,
                                    C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size, ceil(C4_2_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    relu <<< dimGrid4_2, dimBlock4_2 >>> (d_C4_2_feature_map, C4_2_channel, C4_2_size, C4_2_size, ceil(C4_2_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);    
    pad <<< dimGrid4_2, dimBlock4_2 >>> (d_C4_2_feature_map, d_C4_2_feature_map_padded, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size, ceil(C4_2_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);

    dim3 dimGrid4_3(batch, C4_3_channel, ceil((C4_3_size*C4_3_size)/(TILE_WIDTH_QUARTER*TILE_WIDTH_QUARTER))); // batch*256*1
    dim3 dimBlock4_3(TILE_WIDTH_QUARTER,TILE_WIDTH_QUARTER, 1); // 4*4*1
    // conv_shared <<< dimGrid4_3, dimBlock4_3, shmem_size_4 >>> (d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, C4_2_size+2*conv4_3_padding_size,
    //     C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size, ceil(C4_3_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    conv <<< dimGrid4_3, dimBlock4_3 >>> (d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, C4_2_size+2*conv4_3_padding_size,
                                    C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size, ceil(C4_3_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);
    relu <<< dimGrid4_3, dimBlock4_3 >>> (d_C4_3_feature_map, C4_3_channel, C4_3_size, C4_3_size, ceil(C4_3_size/TILE_WIDTH_QUARTER), TILE_WIDTH_QUARTER);

    dim3 dimGrid4_4(batch, S4_channel, ceil((S4_size*S4_size)/(TILE_WIDTH_EIGHTH*TILE_WIDTH_EIGHTH))); // batch*256*1
    dim3 dimBlock4_4(TILE_WIDTH_EIGHTH,TILE_WIDTH_EIGHTH, 1); // 4*4*1
    pool <<< dimGrid4_4, dimBlock4_4 >>> (d_C4_3_feature_map, d_S4_feature_map, S4_channel, S4_size, S4_size, ceil(S4_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);

    //////////BLOCK 5/////////////////////////////////

    dim3 dimGrid5_0(batch, S4_channel, ceil((S4_size*S4_size)/(TILE_WIDTH_EIGHTH*TILE_WIDTH_EIGHTH))); // batch*512*1
    dim3 dimBlock5_0(TILE_WIDTH_EIGHTH,TILE_WIDTH_EIGHTH, 1); // 2*2*1
    pad <<< dimGrid5_0, dimBlock5_0 >>> (d_S4_feature_map, d_S4_feature_map_padded, S4_channel, S4_size, S4_size, conv5_1_padding_size, ceil(S4_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
             
    dim3 dimGrid5_1(batch, C5_1_channel, ceil((C5_1_size*C5_1_size)/(TILE_WIDTH_EIGHTH*TILE_WIDTH_EIGHTH))); // batch*512*1
    dim3 dimBlock5_1(TILE_WIDTH_EIGHTH,TILE_WIDTH_EIGHTH, 1); // 2*2*1
    size_t shmem_size_5 = sizeof(float) * ((TILE_WIDTH_EIGHTH + conv5_1_kernel_size-1)*(TILE_WIDTH_EIGHTH+conv5_1_kernel_size-1) + conv5_1_kernel_size*conv5_1_kernel_size);
    // conv_shared <<< dimGrid5_1, dimBlock5_1, shmem_size_5>>> (d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias, S4_size+2*conv5_1_padding_size,
    //     S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size, ceil(C5_1_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    conv <<< dimGrid5_1, dimBlock5_1 >>> (d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias, S4_size+2*conv5_1_padding_size,
                                    S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size, ceil(C5_1_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    relu <<< dimGrid5_1, dimBlock5_1 >>> (d_C5_1_feature_map, C5_1_channel, C5_1_size, C5_1_size, ceil(C5_1_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    pad <<< dimGrid5_1, dimBlock5_1 >>> (d_C5_1_feature_map, d_C5_1_feature_map_padded, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size, ceil(C5_1_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);

    dim3 dimGrid5_2(batch, C5_2_channel, ceil((C5_2_size*C5_2_size)/(TILE_WIDTH_EIGHTH*TILE_WIDTH_EIGHTH))); // batch*512*1
    dim3 dimBlock5_2(TILE_WIDTH_EIGHTH,TILE_WIDTH_EIGHTH, 1); // 2*2*1
    // conv_shared <<< dimGrid5_2, dimBlock5_2, shmem_size_5 >>> (d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias, C5_1_size+2*conv5_2_padding_size,
    //     C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size, ceil(C5_2_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    conv <<< dimGrid5_2, dimBlock5_2 >>> (d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias, C5_1_size+2*conv5_2_padding_size,
                                    C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size, ceil(C5_2_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    relu <<< dimGrid5_2, dimBlock5_2 >>> (d_C5_2_feature_map, C5_2_channel, C5_2_size, C5_2_size, ceil(C5_2_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    pad <<< dimGrid5_2, dimBlock5_2 >>> (d_C5_2_feature_map, d_C5_2_feature_map_padded, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size, ceil(C5_2_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);

    dim3 dimGrid5_3(batch, C5_3_channel, ceil((C5_3_size*C5_3_size)/(TILE_WIDTH_EIGHTH*TILE_WIDTH_EIGHTH))); // batch*512*1
    dim3 dimBlock5_3(TILE_WIDTH_EIGHTH,TILE_WIDTH_EIGHTH, 1); // 2*2*1
    // conv_shared <<< dimGrid5_3, dimBlock5_3, shmem_size_5 >>> (d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias, C5_2_size+2*conv5_3_padding_size,
    //     C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size, ceil(C5_3_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    conv <<< dimGrid5_3, dimBlock5_3 >>> (d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias, C5_2_size+2*conv5_3_padding_size,
                                    C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size, ceil(C5_3_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);
    relu <<< dimGrid5_3, dimBlock5_3 >>> (d_C5_3_feature_map, C5_3_channel, C5_3_size, C5_3_size, ceil(C5_3_size/TILE_WIDTH_EIGHTH), TILE_WIDTH_EIGHTH);

    dim3 dimGrid5_4(batch, S5_channel, 1); // batch*512*1
    dim3 dimBlock5_4(1,1,1); 
    pool <<< dimGrid5_4, dimBlock5_4 >>> (d_C5_3_feature_map, d_S5_feature_map, S5_channel, S5_size, S5_size, 1, 1);
    
    ////////// fc /////////////////////////////////
    
    dim3 dimGrid6(batch, fc1_out_channel, 1); // batch*10*1
    dim3 dimBlock6(1,1,1); 
    fc <<< dimGrid6, dimBlock6 >>> (d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias, fc1_in_channel, fc1_out_channel);
    // relu <<< dimGrid6, dimBlock6 >>> (d_output, fc1_out_channel, 1, 1, 1);

    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */
}

void vgg16_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size);
  cudaMalloc((void**)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
  cudaMalloc((void**)&d_conv1_2_weight,
             sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                 conv1_2_kernel_size * conv1_2_kernel_size);
  cudaMalloc((void**)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size);
  cudaMalloc((void**)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
  cudaMalloc((void**)&d_conv2_2_weight,
             sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                 conv2_2_kernel_size * conv2_2_kernel_size);
  cudaMalloc((void**)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size);
  cudaMalloc((void**)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
  cudaMalloc((void**)&d_conv3_2_weight,
             sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                 conv3_2_kernel_size * conv3_2_kernel_size);
  cudaMalloc((void**)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
  cudaMalloc((void**)&d_conv3_3_weight,
             sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                 conv3_3_kernel_size * conv3_3_kernel_size);
  cudaMalloc((void**)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size);
  cudaMalloc((void**)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
  cudaMalloc((void**)&d_conv4_2_weight,
             sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                 conv4_2_kernel_size * conv4_2_kernel_size);
  cudaMalloc((void**)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
  cudaMalloc((void**)&d_conv4_3_weight,
             sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                 conv4_3_kernel_size * conv4_3_kernel_size);
  cudaMalloc((void**)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size);
  cudaMalloc((void**)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
  cudaMalloc((void**)&d_conv5_2_weight,
             sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                 conv5_2_kernel_size * conv5_2_kernel_size);
  cudaMalloc((void**)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
  cudaMalloc((void**)&d_conv5_3_weight,
             sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                 conv5_3_kernel_size * conv5_3_kernel_size);
  cudaMalloc((void**)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

  //////////FC 1////////////////////////////////////
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(float) * batch * input_channel * input_size * input_size);

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_input_padded,
             sizeof(float) * batch * input_channel * (input_size+2*conv1_1_padding_size) * (input_size+2*conv1_1_padding_size));
  cudaMalloc((void**)&d_C1_1_feature_map,
             sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
  cudaMalloc((void**)&d_C1_1_feature_map_padded,
             sizeof(float) * batch * C1_1_channel * (C1_1_size+2*conv1_2_padding_size) * (C1_1_size+2*conv1_2_padding_size));
  cudaMalloc((void**)&d_C1_2_feature_map,
             sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
  cudaMalloc((void**)&d_S1_feature_map,
             sizeof(float) * batch * S1_channel * S1_size * S1_size);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_S1_feature_map_padded,
             sizeof(float) * batch * S1_channel * (S1_size+2*conv2_1_padding_size) * (S1_size+2*conv2_1_padding_size));
  cudaMalloc((void**)&d_C2_1_feature_map,
             sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
  cudaMalloc((void**)&d_C2_1_feature_map_padded,
             sizeof(float) * batch * C2_1_channel * (C2_1_size+2*conv2_2_padding_size) * (C2_1_size+2*conv2_2_padding_size));
  cudaMalloc((void**)&d_C2_2_feature_map,
             sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(float) * batch * S2_channel * S2_size * S2_size);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_S2_feature_map_padded,
             sizeof(float) * batch * S2_channel * (S2_size+2*conv3_1_padding_size) * (S2_size+2*conv3_1_padding_size));
  cudaMalloc((void**)&d_C3_1_feature_map,
             sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
  cudaMalloc((void**)&d_C3_1_feature_map_padded,
             sizeof(float) * batch * C3_1_channel * (C3_1_size+2*conv3_2_padding_size) * (C3_1_size+2*conv3_2_padding_size));
  cudaMalloc((void**)&d_C3_2_feature_map,
             sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
  cudaMalloc((void**)&d_C3_2_feature_map_padded,
             sizeof(float) * batch * C3_2_channel * (C3_2_size+2*conv3_3_padding_size) * (C3_2_size+2*conv3_3_padding_size));
  cudaMalloc((void**)&d_C3_3_feature_map,
             sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
  cudaMalloc((void**)&d_S3_feature_map,
             sizeof(float) * batch * S3_channel * S3_size * S3_size);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_S3_feature_map_padded,
             sizeof(float) * batch * S3_channel * (S3_size+2*conv4_1_padding_size) * (S3_size+2*conv4_1_padding_size));
  cudaMalloc((void**)&d_C4_1_feature_map,
             sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
  cudaMalloc((void**)&d_C4_1_feature_map_padded,
             sizeof(float) * batch * C4_1_channel * (C4_1_size+2*conv4_2_padding_size) * (C4_1_size+2*conv4_2_padding_size));
  cudaMalloc((void**)&d_C4_2_feature_map,
             sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
  cudaMalloc((void**)&d_C4_2_feature_map_padded,
             sizeof(float) * batch * C4_2_channel * (C4_2_size+2*conv4_3_padding_size) * (C4_2_size+2*conv4_3_padding_size));
  cudaMalloc((void**)&d_C4_3_feature_map,
             sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(float) * batch * S4_channel * S4_size * S4_size);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_S4_feature_map_padded,
             sizeof(float) * batch * S4_channel * (S4_size+2*conv5_1_padding_size) * (S4_size+2*conv5_1_padding_size));
  cudaMalloc((void**)&d_C5_1_feature_map,
             sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
  cudaMalloc((void**)&d_C5_1_feature_map_padded,
             sizeof(float) * batch * C5_1_channel * (C5_1_size+2*conv5_2_padding_size) * (C5_1_size+2*conv5_2_padding_size));
  cudaMalloc((void**)&d_C5_2_feature_map,
             sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
  cudaMalloc((void**)&d_C5_2_feature_map_padded,
             sizeof(float) * batch * C5_2_channel * (C5_2_size+2*conv5_3_padding_size) * (C5_2_size+2*conv5_3_padding_size));
  cudaMalloc((void**)&d_C5_3_feature_map,
             sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
  cudaMalloc((void**)&d_S5_feature_map,
             sizeof(float) * batch * S5_channel * S5_size * S5_size);


  cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

  // Copy Parameters
  //////////BLOCK 1/////////////////////////////////
  cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
              sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                  conv1_2_kernel_size * conv1_2_kernel_size,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 2/////////////////////////////////
  cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
              sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                  conv2_2_kernel_size * conv2_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 3/////////////////////////////////
  cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
              sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                  conv3_2_kernel_size * conv3_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
              sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                  conv3_3_kernel_size * conv3_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 4/////////////////////////////////
  cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
              sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                  conv4_2_kernel_size * conv4_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
              sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                  conv4_3_kernel_size * conv4_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 5/////////////////////////////////
  cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
              sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                  conv5_2_kernel_size * conv5_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
              sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                  conv5_3_kernel_size * conv5_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel,
              cudaMemcpyHostToDevice);


  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void vgg16_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(float) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda() {
  cudaFree(d_conv1_1_weight);   
  cudaFree(d_conv1_2_weight);   
  cudaFree(d_conv2_1_weight);   
  cudaFree(d_conv2_2_weight);  
  cudaFree(d_conv3_1_weight);   
  cudaFree(d_conv3_2_weight);   
  cudaFree(d_conv3_3_weight);   
  cudaFree(d_conv4_1_weight);   
  cudaFree(d_conv4_2_weight);   
  cudaFree(d_conv4_3_weight); 
  cudaFree(d_conv5_1_weight);   
  cudaFree(d_conv5_2_weight);   
  cudaFree(d_conv5_3_weight);   
 
  cudaFree(d_conv1_1_bias);   
  cudaFree(d_conv1_2_bias);   
  cudaFree(d_conv2_1_bias);   
  cudaFree(d_conv2_2_bias);  
  cudaFree(d_conv3_1_bias);   
  cudaFree(d_conv3_2_bias);   
  cudaFree(d_conv3_3_bias);   
  cudaFree(d_conv4_1_bias);   
  cudaFree(d_conv4_2_bias);   
  cudaFree(d_conv4_3_bias); 
  cudaFree(d_conv5_1_bias);   
  cudaFree(d_conv5_2_bias);   
  cudaFree(d_conv5_3_bias);   
   
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc1_bias);        

  cudaFree(d_image);          
  cudaFree(d_input); 

  cudaFree(d_input_padded);          
  cudaFree(d_C1_1_feature_map); 
  cudaFree(d_C1_1_feature_map_padded); 
  cudaFree(d_C1_2_feature_map); 
  cudaFree(d_S1_feature_map); 

  cudaFree(d_S1_feature_map_padded); 
  cudaFree(d_C2_1_feature_map); 
  cudaFree(d_C2_1_feature_map_padded); 
  cudaFree(d_C2_2_feature_map); 
  cudaFree(d_S2_feature_map); 

  cudaFree(d_S2_feature_map_padded); 
  cudaFree(d_C3_1_feature_map); 
  cudaFree(d_C3_1_feature_map_padded); 
  cudaFree(d_C3_2_feature_map); 
  cudaFree(d_C3_2_feature_map_padded); 
  cudaFree(d_C3_3_feature_map); 
  cudaFree(d_S3_feature_map); 

  cudaFree(d_S3_feature_map_padded); 
  cudaFree(d_C4_1_feature_map); 
  cudaFree(d_C4_1_feature_map_padded); 
  cudaFree(d_C4_2_feature_map); 
  cudaFree(d_C4_2_feature_map_padded); 
  cudaFree(d_C4_3_feature_map); 
  cudaFree(d_S4_feature_map); 

  cudaFree(d_S4_feature_map_padded); 
  cudaFree(d_C5_1_feature_map); 
  cudaFree(d_C5_1_feature_map_padded); 
  cudaFree(d_C5_2_feature_map); 
  cudaFree(d_C5_2_feature_map_padded); 
  cudaFree(d_C5_3_feature_map); 
  cudaFree(d_S5_feature_map); 
 
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
