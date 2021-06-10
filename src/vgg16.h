#ifndef VGG16_H
#define VGG16_H

#include "common.h"
#include <limits>
#include <cmath>
#include <algorithm>

class vgg16
{
public:
    vgg16(int batch = 1);
    ~vgg16();
    virtual void load_parameters(std::string value_path);
    virtual void print_parameters();
    virtual bool compare(vgg16* other);
    virtual void predict(const uint8_t* const image, int batch) = 0;
    virtual void classify(int* predict, int batch) = 0;
protected:
    void softmax(float* input, int* output, int B, int size);
    //////////////////////////////////////////////////
    // Internal parameter
    //////////////////////////////////////////////////
    int batch = 1;
    int parameter_initialized = false;
    //////////////////////////////////////////////////
    // Model Parameters
    //////////////////////////////////////////////////
    float* conv1_1_weight;   // [3][64][3][3];
    float* conv1_2_weight;   // [64][64][3][3];
    float* conv2_1_weight;   // [64][128][3][3];
    float* conv2_2_weight;   // [128][128][3][3];
    float* conv3_1_weight;   // [128][256][3][3];
    float* conv3_2_weight;   // [256][256][3][3];
    float* conv3_3_weight;   // [256][256][3][3];
    float* conv4_1_weight;   // [256][512][3][3];
    float* conv4_2_weight;   // [512][512][3][3];
    float* conv4_3_weight;   // [512][512][3][3];
    float* conv5_1_weight;   // [512][512][3][3];
    float* conv5_2_weight;   // [512][512][3][3];
    float* conv5_3_weight;   // [512][512][3][3];

    float* conv1_1_bias;   // [64];
    float* conv1_2_bias;   // [64];
    float* conv2_1_bias;   // [128];
    float* conv2_2_bias;   // [128];
    float* conv3_1_bias;   // [256];
    float* conv3_2_bias;   // [256];
    float* conv3_3_bias;   // [256];
    float* conv4_1_bias;   // [512];
    float* conv4_2_bias;   // [512];
    float* conv4_3_bias;   // [512];
    float* conv5_1_bias;   // [512];
    float* conv5_2_bias;   // [512];
    float* conv5_3_bias;   // [512];

    float* fc1_weight;     // [512][10];
    float* fc1_bias;       // [10];
    //////////////////////////////////////////////////
    // Feature Map
    //////////////////////////////////////////////////
    float* input;                   // [batch][3][32][32];

    float* input_padded;             // [batch][3][34][34]
    float* C1_1_feature_map;        // [batch][64][32][32];
    float* C1_1_feature_map_padded; // [batch][64][34][34];
    float* C1_2_feature_map;        // [batch][64][32][32];
    float* S1_feature_map;          // [batch][64][16][16];

    float* S1_feature_map_padded;   // [batch][64][18][18];
    float* C2_1_feature_map;        // [batch][128][16][16];
    float* C2_1_feature_map_padded; // [batch][128][18][18]
    float* C2_2_feature_map;        // [batch][128][16][16];
    float* S2_feature_map;          // [batch][128][8][8];

    float* S2_feature_map_padded;   // [batch][128][10][10];
    float* C3_1_feature_map;        // [batch][256][8][8];
    float* C3_1_feature_map_padded; // [batch][256][10][10];
    float* C3_2_feature_map;        // [batch][256][8][8];
    float* C3_2_feature_map_padded; // [batch][256][10][10];
    float* C3_3_feature_map;        // [batch][256][8][8];
    float* S3_feature_map;          // [batch][256][4][4];

    float* S3_feature_map_padded;   // [batch][256][6][6];
    float* C4_1_feature_map;        // [batch][512][4][4];
    float* C4_1_feature_map_padded; // [batch][512][6][6];
    float* C4_2_feature_map;        // [batch][512][4][4];
    float* C4_2_feature_map_padded; // [batch][512][6][6];
    float* C4_3_feature_map;        // [batch][512][4][4];
    float* S4_feature_map;          // [batch][512][2][2];

    float* S4_feature_map_padded;   // [batch][512][4][4];
    float* C5_1_feature_map;        // [batch][512][2][2];
    float* C5_1_feature_map_padded; // [batch][512][4][4];
    float* C5_2_feature_map;        // [batch][512][2][2];
    float* C5_2_feature_map_padded; // [batch][512][4][4];
    float* C5_3_feature_map;        // [batch][512][2][2];
    float* S5_feature_map;          // [batch][512][1][1];

    float* output;                  // [batch][10];
    //////////////////////////////////////////////////
    // Layer and Feature map parameters
    //     Check README.md for more information
    //////////////////////////////////////////////////
    //// Input
    int input_size = 32;
    int input_channel = 3;

    //////////BLOCK 1/////////////////////////////////
    //// [Layer] conv1_1
    int conv1_1_in_channel = 3;
    int conv1_1_out_channel = 64;
    int conv1_1_kernel_size = 3;
    int conv1_1_padding_size = 1;
    //// C1_1 feature map
    int C1_1_channel = conv1_1_out_channel;
    int C1_1_size = input_size - (conv1_1_kernel_size - 1) + 2*conv1_1_padding_size;

    //// [Layer] conv1_2
    int conv1_2_in_channel = C1_1_channel;
    int conv1_2_out_channel = 64;
    int conv1_2_kernel_size = 3;
    int conv1_2_padding_size = 1;
    //// C1_2 feature map
    int C1_2_channel = conv1_2_out_channel;
    int C1_2_size = C1_1_size - (conv1_2_kernel_size - 1) + 2*conv1_2_padding_size;

    //// S1 feature map
    int S1_channel = C1_2_channel;
    int S1_size = C1_2_size / 2;

    //////////BLOCK 2/////////////////////////////////
    //// [Layer] conv2_1
    int conv2_1_in_channel = S1_channel;
    int conv2_1_out_channel = 128;
    int conv2_1_kernel_size = 3;
    int conv2_1_padding_size = 1;
    //// C2_1 feature map
    int C2_1_channel = conv2_1_out_channel;
    int C2_1_size = S1_size - (conv2_1_kernel_size - 1) + 2*conv2_1_padding_size;

    //// [Layer] conv2_2
    int conv2_2_in_channel = C2_1_channel;
    int conv2_2_out_channel = 128;
    int conv2_2_kernel_size = 3;
    int conv2_2_padding_size = 1;
    //// C2_2 feature map
    int C2_2_channel = conv2_2_out_channel;
    int C2_2_size = C2_1_size - (conv2_2_kernel_size - 1) + 2*conv2_2_padding_size;

    //// S2 feature map
    int S2_channel = C2_2_channel;
    int S2_size = C2_2_size / 2;

    //////////BLOCK 3/////////////////////////////////
    //// [Layer] conv3_1
    int conv3_1_in_channel = S2_channel;
    int conv3_1_out_channel = 256;
    int conv3_1_kernel_size = 3;
    int conv3_1_padding_size = 1;
    //// C3_1 feature map
    int C3_1_channel = conv3_1_out_channel;
    int C3_1_size = S2_size - (conv3_1_kernel_size - 1) + 2*conv3_1_padding_size;

    //// [Layer] conv3_2
    int conv3_2_in_channel = C3_1_channel;
    int conv3_2_out_channel = 256;
    int conv3_2_kernel_size = 3;
    int conv3_2_padding_size = 1;
    //// C3_2 feature map
    int C3_2_channel = conv3_2_out_channel;
    int C3_2_size = C3_1_size - (conv3_2_kernel_size - 1) + 2*conv3_2_padding_size;

    //// [Layer] conv3_3
    int conv3_3_in_channel = C3_2_channel;
    int conv3_3_out_channel = 256;
    int conv3_3_kernel_size = 3;
    int conv3_3_padding_size = 1;
    //// C3_3 feature map
    int C3_3_channel = conv3_3_out_channel;
    int C3_3_size = C3_2_size - (conv3_3_kernel_size - 1) + 2*conv3_3_padding_size;

    //// S4 feature map
    int S3_channel = C3_3_channel;
    int S3_size = C3_3_size / 2;

    //////////BLOCK 4/////////////////////////////////
    //// [Layer] conv4_1
    int conv4_1_in_channel = S3_channel;
    int conv4_1_out_channel = 512;
    int conv4_1_kernel_size = 3;
    int conv4_1_padding_size = 1;
    //// C4_1 feature map
    int C4_1_channel = conv4_1_out_channel;
    int C4_1_size = S3_size - (conv4_1_kernel_size - 1) + 2*conv4_1_padding_size;

    //// [Layer] conv4_2
    int conv4_2_in_channel = C4_1_channel;
    int conv4_2_out_channel = 512;
    int conv4_2_kernel_size = 3;
    int conv4_2_padding_size = 1;
    //// C4_2 feature map
    int C4_2_channel = conv4_2_out_channel;
    int C4_2_size = C4_1_size - (conv4_2_kernel_size - 1) + 2*conv4_2_padding_size;

    //// [Layer] conv4_3
    int conv4_3_in_channel = C4_2_channel;
    int conv4_3_out_channel = 512;
    int conv4_3_kernel_size = 3;
    int conv4_3_padding_size = 1;
    //// C4_3 feature map
    int C4_3_channel = conv4_3_out_channel;
    int C4_3_size = C4_2_size - (conv4_3_kernel_size - 1) + 2*conv4_3_padding_size;

    //// S4 feature map
    int S4_channel = C4_3_channel;
    int S4_size = C4_3_size / 2;

    //////////BLOCK 5/////////////////////////////////
    //// [Layer] conv5_1
    int conv5_1_in_channel = S4_channel;
    int conv5_1_out_channel = 512;
    int conv5_1_kernel_size = 3;
    int conv5_1_padding_size = 1;
    //// C5_1 feature map
    int C5_1_channel = conv5_1_out_channel;
    int C5_1_size = S4_size - (conv5_1_kernel_size - 1) + 2*conv5_1_padding_size;

    //// [Layer] conv5_2
    int conv5_2_in_channel = C5_1_channel;
    int conv5_2_out_channel = 512;
    int conv5_2_kernel_size = 3;
    int conv5_2_padding_size = 1;
    //// C5_2 feature map
    int C5_2_channel = conv5_2_out_channel;
    int C5_2_size = C5_1_size - (conv5_2_kernel_size - 1) + 2*conv5_2_padding_size;

    //// [Layer] conv5_3
    int conv5_3_in_channel = C5_2_channel;
    int conv5_3_out_channel = 512;
    int conv5_3_kernel_size = 3;
    int conv5_3_padding_size = 1;
    //// C5_3 feature map
    int C5_3_channel = conv5_3_out_channel;
    int C5_3_size = C5_2_size - (conv5_3_kernel_size - 1) + 2*conv5_3_padding_size;

    //// S5 feature map
    int S5_channel = C5_3_channel;
    int S5_size = C5_3_size / 2;


    //// [Layer] fc1
    int fc1_in_channel = S5_channel * S5_size * S5_size;
    int fc1_out_channel = 10;

    //// output
    int output_size = fc1_out_channel;
};

#endif
