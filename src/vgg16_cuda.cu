#include "vgg16_cuda.h"
#define TILE_WIDTH 16
#define FLT_MIN -3.40282e+38
#define FLT_EPSILON 1.19209e-07

// N: minibatch size
// M: output feature map size
// C: number of channels
// H,W: input height width
// P: pad size
// input, output, weight 

__global__ void pad(float* input, float* output, int C, int H, int W, int P, int W_grid) {

  int b = blockIdx.x; // mini_batch
  int c = blockIdx.y; // input channel
  int h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y; // input height
  int w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x; // input width 

  int H_OUT = H+2*P;
  int W_OUT = W+2*P;

  int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
  int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h+P) * W_OUT + w+P;

  output[output_index] = input[input_base];
}

__global__ void conv(float* input, float* output, float* weight, float* bias, int H, int W, int IC, int OC, int K, int W_grid)
{
    int b = blockIdx.x; // mini_batch
    int oc = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x; // output width

    int H_OUT = H - (K - 1);
    int W_OUT = W - (K - 1);
    int output_index = b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;

    float val = 0;
    for (int ic=0; ic<IC; ic++){
        int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
        int kernel_base = oc * (IC * K * K) + ic * (K * K);
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++) {
                val += input[input_base + kh * (W) + kw] * weight[kernel_base + kh * (K) + kw];
            }
        }

    output[output_index] = bias[oc];
    output[output_index] += val; 
}

__global__ void relu(float* input, int C, int H, int W, int W_grid)
{
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y; // input height
    int w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x; // input width 

    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
    input[input_base] = max(input[input_base], (float)(0.0));
    
}

__global__ void pool(float* input, float* output, int C, int H, int W, int W_grid)
{   
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*TILE_WIDTH + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*TILE_WIDTH + threadIdx.x; // output width 

    int scale = 2;
    int H_OUT = H / scale;
    int W_OUT = W / scale;

    int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
    int max_sh = 0; 
    int max_sw = 0;
    float max_val = FLT_MIN;
    
    // Find maximum
    for (int sh = 0; sh < scale; sh++)
        for (int sw = 0; sw < scale; sw++) {
        float val = input[input_base + sh * (W) + sw];
        if (val - max_val > FLT_EPSILON) {
            max_val = val;
            max_sh = sh;
            max_sw = sw;
        }
    }
    int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h / 2) * W_OUT + (w / 2);
    output[output_index] = max_val;
}

__global__ void fc(float* input, float* output, float* weight, float* bias, int IC, int OC)
{

}

void vgg16_cuda::predict(int batch) {

    int n_channel=0; int n_tile=0;
    dim3 dimGrid(batch, n_channel, n_tile); // (minibatch) * (# of feature map) * (# of tiles) 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1); // (tile_width * tile_width * 1)
    //////////BLOCK 1/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv1_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv1_2
    // TODO: Implement relu
    // TODO: Implement pool

    n_channel = input_channel; n_tile = (input_size*input_size)/(TILE_WIDTH*TILE_WIDTH);
    pad <<< dimGrid, dimBlock >>> (d_input, d_input_padded, input_channel, input_size, input_size, conv1_1_padding_size, input_size/TILE_WIDTH);

    n_channel = conv1_1_out_channel; n_tile = (C1_1_size*C1_1_size)/(TILE_WIDTH*TILE_WIDTH);
    conv <<< dimGrid, dimBlock >>> (d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, input_size+2*conv1_1_padding_size,
                                    input_size+2*conv1_1_padding_size, conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size, C1_1_size/TILE_WIDTH);
    relu <<< dimGrid, dimBlock >>> (d_C1_1_feature_map, C1_1_channel, C1_1_size, C1_1_size, C1_1_size/TILE_WIDTH);

    n_channel = conv1_2_in_channel; n_tile = (C1_1_size*C1_1_size)/(TILE_WIDTH*TILE_WIDTH);
    pad <<< dimGrid, dimBlock >>> (d_C1_1_feature_map, d_C1_1_feature_map_padded, C1_1_channel, C1_1_size, C1_1_size, conv1_2_padding_size, C1_1_size/TILE_WIDTH);

    n_channel = conv1_2_out_channel; n_tile = (C1_2_size*C1_2_size)/(TILE_WIDTH*TILE_WIDTH);
    conv <<< dimGrid, dimBlock >>> (d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, C1_1_size+2*conv1_2_padding_size,
                                    C1_1_size+2*conv1_2_padding_size, conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size, C1_2_size/TILE_WIDTH);
    relu <<< dimGrid, dimBlock >>> (d_C1_2_feature_map, C1_2_channel, C1_2_size, C1_2_size, C1_2_size/TILE_WIDTH);

    n_channel = S1_channel; n_tile = (S1_size*S1_size)/(TILE_WIDTH*TILE_WIDTH);
    pool <<< dimGrid, dimBlock >>> (d_C1_2_feature_map, d_S1_feature_map, S1_channel, C1_2_size, C1_2_size, C1_2_size/TILE_WIDTH);
    
    // return;
    // //////////BLOCK 2/////////////////////////////////
    // // TODO: Implement pad
    // // TODO: Implement conv2_1
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv2_2
    // // TODO: Implement relu
    // // TODO: Implement pool

    // pad <<< dimGrid, dimBlock >>> (S1_feature_map, S1_feature_map_padded, S1_channel, S1_size, S1_size, conv2_1_padding_size);
    // conv <<< dimGrid, dimBlock >>> (S1_feature_map_padded, C2_1_feature_map, conv2_1_weight, conv2_1_bias, S1_size+2*conv2_1_padding_size,
    //     S1_size+2*conv2_1_padding_size, conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C2_1_feature_map, C2_1_channel * C2_1_size * C2_1_size);
    // pad  <<< dimGrid, dimBlock >>> (C2_1_feature_map, C2_1_feature_map_padded, C2_1_channel, C2_1_size, C2_1_size, conv2_2_padding_size);
    // conv <<< dimGrid, dimBlock >>> (C2_1_feature_map_padded, C2_2_feature_map, conv2_2_weight, conv2_2_bias, C2_1_size+2*conv2_2_padding_size,
    //     C2_1_size+2*conv2_2_padding_size, conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C2_2_feature_map, C2_2_channel * C2_2_size * C2_2_size);
    // pool <<< dimGrid, dimBlock >>> (C2_2_feature_map, S2_feature_map, C2_2_channel, C2_2_size, C2_2_size);

    // //////////BLOCK 3/////////////////////////////////
    // // TODO: Implement pad
    // // TODO: Implement conv3_1
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv3_2
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv3_3
    // // TODO: Implement relu
    // // TODO: Implement pool
    // pad <<< dimGrid, dimBlock >>> (S2_feature_map, S2_feature_map_padded, S2_channel, S2_size, S2_size, conv3_1_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (S2_feature_map_padded, C3_1_feature_map, conv3_1_weight, conv3_1_bias, S2_size+2*conv3_1_padding_size,
    //     S2_size+2*conv3_1_padding_size, conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C3_1_feature_map, C3_1_channel * C3_1_size * C3_1_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C3_1_feature_map, C3_1_feature_map_padded, C3_1_channel, C3_1_size, C3_1_size, conv3_2_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C3_1_feature_map_padded, C3_2_feature_map, conv3_2_weight, conv3_2_bias, C3_1_size+2*conv3_2_padding_size,
    //     C3_1_size+2*conv3_2_padding_size, conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C3_2_feature_map, C3_2_channel * C3_2_size * C3_2_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C3_2_feature_map, C3_2_feature_map_padded, C3_2_channel, C3_2_size, C3_2_size, conv3_3_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C3_2_feature_map_padded, C3_3_feature_map, conv3_3_weight, conv3_3_bias, C3_2_size+2*conv3_3_padding_size,
    //     C3_2_size+2*conv3_3_padding_size, conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C3_3_feature_map, C3_3_channel * C3_3_size * C3_3_size);
    // // MaxPool2d
    // pool <<< dimGrid, dimBlock >>> (C3_3_feature_map, S3_feature_map, C3_3_channel, C3_3_size, C3_3_size);

    // //////////BLOCK 4/////////////////////////////////
    // // TODO: Implement pad
    // // TODO: Implement conv4_1
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv4_2
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv4_3
    // // TODO: Implement relu
    // // TODO: Implement pool
    // pad <<< dimGrid, dimBlock >>> (S3_feature_map, S3_feature_map_padded, S3_channel, S3_size, S3_size, conv4_1_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (S3_feature_map_padded, C4_1_feature_map, conv4_1_weight, conv4_1_bias, S3_size+2*conv4_1_padding_size,
    //     S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C4_1_feature_map, C4_1_channel * C4_1_size * C4_1_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C4_1_feature_map, C4_1_feature_map_padded, C4_1_channel, C4_1_size, C4_1_size, conv4_2_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C4_1_feature_map_padded, C4_2_feature_map, conv4_2_weight, conv4_2_bias, C4_1_size+2*conv4_2_padding_size,
    //     C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C4_2_feature_map, C4_2_channel * C4_2_size * C4_2_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C4_2_feature_map, C4_2_feature_map_padded, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C4_2_feature_map_padded, C4_3_feature_map, conv4_3_weight, conv4_3_bias, C4_2_size+2*conv4_3_padding_size,
    //     C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C4_3_feature_map, C4_3_channel * C4_3_size * C4_3_size);
    // // MaxPool2d
    // pool <<< dimGrid, dimBlock >>> (C4_3_feature_map, S4_feature_map, C4_3_channel, C4_3_size, C4_3_size);

    // //////////BLOCK 5/////////////////////////////////
    // // TODO: Implement pad
    // // TODO: Implement conv5_1
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv5_2
    // // TODO: Implement relu
    // // TODO: Implement pad
    // // TODO: Implement conv5_3
    // // TODO: Implement relu
    // // TODO: Implement pool
    // pad <<< dimGrid, dimBlock >>> (S4_feature_map, S4_feature_map_padded, S4_channel, S4_size, S4_size, conv5_1_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (S4_feature_map_padded, C5_1_feature_map, conv5_1_weight, conv5_1_bias, S4_size+2*conv5_1_padding_size,
    //     S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C5_1_feature_map, C5_1_channel * C5_1_size * C5_1_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C5_1_feature_map, C5_1_feature_map_padded, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C5_1_feature_map_padded, C5_2_feature_map, conv5_2_weight, conv5_2_bias, C5_1_size+2*conv5_2_padding_size,
    //     C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C5_2_feature_map, C5_2_channel * C5_2_size * C5_2_size);
    // // ZeroPad2d
    // pad <<< dimGrid, dimBlock >>> (C5_2_feature_map, C5_2_feature_map_padded, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size);
    // // conv2d
    // conv <<< dimGrid, dimBlock >>> (C5_2_feature_map_padded, C5_3_feature_map, conv5_3_weight, conv5_3_bias, C5_2_size+2*conv5_3_padding_size,
    //     C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);
    // relu <<< dimGrid, dimBlock >>> (C5_3_feature_map, C5_3_channel * C5_3_size * C5_3_size);
    // // MaxPool2d
    // pool <<< dimGrid, dimBlock >>> (C5_3_feature_map, S5_feature_map, C5_3_channel, C5_3_size, C5_3_size);
    // // Linear
    
    // // TODO: Implement fc1
    // // TODO: Implement relu
    // // dimGrid(batch, fc1_out_channel, TILE_WIDTH); 
    // // dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // fc <<< dimGrid, dimBlock >>> (S5_feature_map, output, fc1_weight, fc1_bias, fc1_in_channel, fc1_out_channel);
    

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
