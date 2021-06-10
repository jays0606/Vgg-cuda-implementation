#include "vgg16_cpu.h"

void vgg16_cpu::predict(const uint8_t* const image, int batch) {
  // ToTensor and Normalize
  normalize(image, input);

  //////////BLOCK 1/////////////////////////////////
  // ZeroPad2d
  pad(input, input_padded, batch, input_channel, input_size, input_size, conv1_1_padding_size);
  // Conv2d
  conv(input_padded, C1_1_feature_map, conv1_1_weight, conv1_1_bias, batch, input_size+2*conv1_1_padding_size,
       input_size+2*conv1_1_padding_size, conv1_1_in_channel, conv1_1_out_channel, conv1_1_kernel_size);

  relu(C1_1_feature_map, batch * C1_1_channel * C1_1_size * C1_1_size);
  // ZeroPad2d
  pad(C1_1_feature_map, C1_1_feature_map_padded, batch, C1_1_channel, C1_1_size, C1_1_size, conv1_2_padding_size);
  // Conv2d
  conv(C1_1_feature_map_padded, C1_2_feature_map, conv1_2_weight, conv1_2_bias, batch, C1_1_size+2*conv1_2_padding_size,
       C1_1_size+2*conv1_2_padding_size, conv1_2_in_channel, conv1_2_out_channel, conv1_2_kernel_size);

  relu(C1_2_feature_map, batch * C1_2_channel * C1_2_size * C1_2_size);

  // MaxPool2d
  pool(C1_2_feature_map, S1_feature_map, batch, C1_2_channel, C1_2_size, C1_2_size);

  //////////BLOCK 2/////////////////////////////////
  // ZeroPad2d
  pad(S1_feature_map, S1_feature_map_padded, batch, S1_channel, S1_size, S1_size, conv2_1_padding_size);
  // Conv2d
  conv(S1_feature_map_padded, C2_1_feature_map, conv2_1_weight, conv2_1_bias, batch, S1_size+2*conv2_1_padding_size,
       S1_size+2*conv2_1_padding_size, conv2_1_in_channel, conv2_1_out_channel, conv2_1_kernel_size);
  relu(C2_1_feature_map, batch * C2_1_channel * C2_1_size * C2_1_size);
  // ZeroPad2d
  pad(C2_1_feature_map, C2_1_feature_map_padded, batch, C2_1_channel, C2_1_size, C2_1_size, conv2_2_padding_size);
  // Conv2d
  conv(C2_1_feature_map_padded, C2_2_feature_map, conv2_2_weight, conv2_2_bias, batch, C2_1_size+2*conv2_2_padding_size,
       C2_1_size+2*conv2_2_padding_size, conv2_2_in_channel, conv2_2_out_channel, conv2_2_kernel_size);
  relu(C2_2_feature_map, batch * C2_2_channel * C2_2_size * C2_2_size);
  // MaxPool2d
  pool(C2_2_feature_map, S2_feature_map, batch, C2_2_channel, C2_2_size, C2_2_size);

  //////////BLOCK 3/////////////////////////////////
  // ZeroPad2d
  pad(S2_feature_map, S2_feature_map_padded, batch, S2_channel, S2_size, S2_size, conv3_1_padding_size);
  // conv2d
  conv(S2_feature_map_padded, C3_1_feature_map, conv3_1_weight, conv3_1_bias, batch, S2_size+2*conv3_1_padding_size,
       S2_size+2*conv3_1_padding_size, conv3_1_in_channel, conv3_1_out_channel, conv3_1_kernel_size);
  relu(C3_1_feature_map, batch * C3_1_channel * C3_1_size * C3_1_size);
  // ZeroPad2d
  pad(C3_1_feature_map, C3_1_feature_map_padded, batch, C3_1_channel, C3_1_size, C3_1_size, conv3_2_padding_size);
  // conv2d
  conv(C3_1_feature_map_padded, C3_2_feature_map, conv3_2_weight, conv3_2_bias, batch, C3_1_size+2*conv3_2_padding_size,
       C3_1_size+2*conv3_2_padding_size, conv3_2_in_channel, conv3_2_out_channel, conv3_2_kernel_size);
  relu(C3_2_feature_map, batch * C3_2_channel * C3_2_size * C3_2_size);
  // ZeroPad2d
  pad(C3_2_feature_map, C3_2_feature_map_padded, batch, C3_2_channel, C3_2_size, C3_2_size, conv3_3_padding_size);
  // conv2d
  conv(C3_2_feature_map_padded, C3_3_feature_map, conv3_3_weight, conv3_3_bias, batch, C3_2_size+2*conv3_3_padding_size,
       C3_2_size+2*conv3_3_padding_size, conv3_3_in_channel, conv3_3_out_channel, conv3_3_kernel_size);
  relu(C3_3_feature_map, batch * C3_3_channel * C3_3_size * C3_3_size);
  // MaxPool2d
  pool(C3_3_feature_map, S3_feature_map, batch, C3_3_channel, C3_3_size, C3_3_size);

  //////////BLOCK 4/////////////////////////////////
  // ZeroPad2d
  pad(S3_feature_map, S3_feature_map_padded, batch, S3_channel, S3_size, S3_size, conv4_1_padding_size);
  // conv2d
  conv(S3_feature_map_padded, C4_1_feature_map, conv4_1_weight, conv4_1_bias, batch, S3_size+2*conv4_1_padding_size,
       S3_size+2*conv4_1_padding_size, conv4_1_in_channel, conv4_1_out_channel, conv4_1_kernel_size);
  relu(C4_1_feature_map, batch * C4_1_channel * C4_1_size * C4_1_size);
  // ZeroPad2d
  pad(C4_1_feature_map, C4_1_feature_map_padded, batch, C4_1_channel, C4_1_size, C4_1_size, conv4_2_padding_size);
  // conv2d
  conv(C4_1_feature_map_padded, C4_2_feature_map, conv4_2_weight, conv4_2_bias, batch, C4_1_size+2*conv4_2_padding_size,
       C4_1_size+2*conv4_2_padding_size, conv4_2_in_channel, conv4_2_out_channel, conv4_2_kernel_size);
  relu(C4_2_feature_map, batch * C4_2_channel * C4_2_size * C4_2_size);
  // ZeroPad2d
  pad(C4_2_feature_map, C4_2_feature_map_padded, batch, C4_2_channel, C4_2_size, C4_2_size, conv4_3_padding_size);
  // conv2d
  conv(C4_2_feature_map_padded, C4_3_feature_map, conv4_3_weight, conv4_3_bias, batch, C4_2_size+2*conv4_3_padding_size,
       C4_2_size+2*conv4_3_padding_size, conv4_3_in_channel, conv4_3_out_channel, conv4_3_kernel_size);
  relu(C4_3_feature_map, batch * C4_3_channel * C4_3_size * C4_3_size);
  // MaxPool2d
  pool(C4_3_feature_map, S4_feature_map, batch, C4_3_channel, C4_3_size, C4_3_size);

  //////////BLOCK 5/////////////////////////////////
  // ZeroPad2d
  pad(S4_feature_map, S4_feature_map_padded, batch, S4_channel, S4_size, S4_size, conv5_1_padding_size);
  // conv2d
  conv(S4_feature_map_padded, C5_1_feature_map, conv5_1_weight, conv5_1_bias, batch, S4_size+2*conv5_1_padding_size,
       S4_size+2*conv5_1_padding_size, conv5_1_in_channel, conv5_1_out_channel, conv5_1_kernel_size);
  relu(C5_1_feature_map, batch * C5_1_channel * C5_1_size * C5_1_size);
  // ZeroPad2d
  pad(C5_1_feature_map, C5_1_feature_map_padded, batch, C5_1_channel, C5_1_size, C5_1_size, conv5_2_padding_size);
  // conv2d
  conv(C5_1_feature_map_padded, C5_2_feature_map, conv5_2_weight, conv5_2_bias, batch, C5_1_size+2*conv5_2_padding_size,
       C5_1_size+2*conv5_2_padding_size, conv5_2_in_channel, conv5_2_out_channel, conv5_2_kernel_size);
  relu(C5_2_feature_map, batch * C5_2_channel * C5_2_size * C5_2_size);
  // ZeroPad2d
  pad(C5_2_feature_map, C5_2_feature_map_padded, batch, C5_2_channel, C5_2_size, C5_2_size, conv5_3_padding_size);
  // conv2d
  conv(C5_2_feature_map_padded, C5_3_feature_map, conv5_3_weight, conv5_3_bias, batch, C5_2_size+2*conv5_3_padding_size,
       C5_2_size+2*conv5_3_padding_size, conv5_3_in_channel, conv5_3_out_channel, conv5_3_kernel_size);
  relu(C5_3_feature_map, batch * C5_3_channel * C5_3_size * C5_3_size);
  // MaxPool2d
  pool(C5_3_feature_map, S5_feature_map, batch, C5_3_channel, C5_3_size, C5_3_size);
  // Linear
  fc(S5_feature_map, output, fc1_weight, fc1_bias, batch, fc1_in_channel,
     fc1_out_channel);
}

void vgg16_cpu::normalize(const uint8_t* const image, float* input) {
  // Initialize variables
  float max_int = 255.0L;
  float mean = 0.5L;
  float var = 0.5L;
  // Normalize
  for (int i = 0; i < batch * input_channel * input_size * input_size; i++) {
    input[i] = image[i] / max_int;       // transforms.ToTensor();
    input[i] = (input[i] - mean) / var;  // transforms.Normalize();
  }
}

void vgg16_cpu::relu(float* feature_map, int size) {
  // relu
  for (int i = 0; i < size; i++) feature_map[i] = std::max(feature_map[i], (float)0.0);
}

void vgg16_cpu::pad(float* input, float* input_padded, int B, int C, int H, int W, int P) {
  int H_OUT = H+2*P;
  int W_OUT = W+2*P;
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++) {
          // Init values
          int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;

          // Set output with max value
          int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                             (h+P) * W_OUT + (w + P);

          input_padded[output_index] = input[input_base];
        }
}

void vgg16_cpu::conv(float* input, float* output, float* weight,
                      float* bias, int B, int H, int W, int IC, int OC,
                      int K) {
  // Initialize variable
  int H_OUT = H - (K - 1);
  int W_OUT = W - (K - 1);
  // Convolution
  for (int b = 0; b < B; b++)              // mini-batch
    for (int oc = 0; oc < OC; oc++) {      // Output Channel
      for (int h = 0; h < H_OUT; h++)      // Height
        for (int w = 0; w < W_OUT; w++) {  // Width
          int output_index =
              b * (OC * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
          output[output_index] = bias[oc];
          for (int ic = 0; ic < IC; ic++) {
            int input_base = b * (IC * H * W) + ic * (H * W) + h * (W) + w;
            int kernel_base = oc * (IC * K * K) + ic * (K * K);
            for (int kh = 0; kh < K; kh++)
              for (int kw = 0; kw < K; kw++) {
                float val = input[input_base + kh * (W) + kw] *
                             weight[kernel_base + kh * (K) + kw];
                output[output_index] += val;
              }
          }
        }
    }
}

void vgg16_cpu::pool(float* input, float* output, int B, int C, int H,
                      int W) {
  // Initilaize variable
  int scale = 2;
  int H_OUT = H / scale;
  int W_OUT = W / scale;
  // Max Pooling
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < H; h += 2)
        for (int w = 0; w < W; w += 2) {
          // Init values
          int input_base = b * (C * H * W) + c * (H * W) + h * (W) + w;
          int max_sh = 0;
          int max_sw = 0;
          float max_val = std::numeric_limits<float>::lowest();
          // Find maximum
          for (int sh = 0; sh < scale; sh++)
            for (int sw = 0; sw < scale; sw++) {
              float val = input[input_base + sh * (W) + sw];
              if (val - max_val > std::numeric_limits<float>::epsilon()) {
                max_val = val;
                max_sh = sh;
                max_sw = sw;
              }
            }
          // Set output with max value
          int output_index = b * (C * H_OUT * W_OUT) + c * (H_OUT * W_OUT) +
                             (h / 2) * W_OUT + (w / 2);
          output[output_index] = max_val;
        }
}

void vgg16_cpu::fc(float* input, float* output, float* weight, float* bias,
                    int B, int IC, int OC) {
  // Fully Connected
  for (int b = 0; b < B; b++)
    for (int oc = 0; oc < OC; oc++) {
      output[b * OC + oc] = bias[oc];
      for (int ic = 0; ic < IC; ic++)
        output[b * OC + oc] += weight[oc * IC + ic] * input[b * IC + ic];
    }
}

void vgg16_cpu::classify(int* predict, int batch) {
  // Softmax
  softmax(output, predict, batch, output_size);
}

void vgg16_cpu::print_fc(float* data, int size) {
  printf("[DEBUG] print %d\n", size);
  for (int i = 0; i < size; i++) printf("%lf\n", data[i]);
}

void vgg16_cpu::print_C1() {
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < C1_1_channel; c++) {
      for (int h = 0; h < C1_1_size; h++) {
        for (int w = 0; w < C1_1_size; w++) {
          printf("%lf ",
                 C1_1_feature_map[b * (C1_1_channel * C1_1_size * C1_1_size) +
                                c * (C1_1_size * C1_1_size) + h * (C1_1_size) + w]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}

void vgg16_cpu::print_C3() {
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < C3_1_channel; c++) {
      for (int h = 0; h < C3_1_size; h++) {
        for (int w = 0; w < C3_1_size; w++) {
          printf("%lf ",
                 C3_1_feature_map[b * (C3_1_channel * C3_1_size * C3_1_size) +
                                c * (C3_1_size * C3_1_size) + h * (C3_1_size) + w]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}
