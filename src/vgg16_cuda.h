#ifndef VGG16_CUDA_H
#define VGG16_CUDA_H

#include "vgg16.h"

class vgg16_cuda : public vgg16
{
public:
    // Get from base class
    void load_parameters(std::string value_path) override { vgg16::load_parameters(value_path); };
    void print_parameters() override { vgg16::print_parameters(); };
    bool compare(vgg16* other) override { return vgg16::compare(other); };
    void prepare_device_memory(uint8_t* image); 
    // Implement!
    vgg16_cuda(int batch = 1) : vgg16(batch) {};
    void predict(int batch) ;
    void predict(const uint8_t* const image, int batch) override {predict(batch);}
    void classify(int* predict, int batch) override;
    ~vgg16_cuda();
private:
    //////////////////////////////////////////////////
    //Device Weights 
    //////////////////////////////////////////////////
    float* d_conv1_1_weight;   // [3][64][3][3];
    float* d_conv1_2_weight;   // [64][64][3][3];
    float* d_conv2_1_weight;   // [64][128][3][3];
    float* d_conv2_2_weight;   // [128][128][3][3];
    float* d_conv3_1_weight;   // [128][256][3][3];
    float* d_conv3_2_weight;   // [256][256][3][3];
    float* d_conv3_3_weight;   // [256][256][3][3];
    float* d_conv4_1_weight;   // [256][512][3][3];
    float* d_conv4_2_weight;   // [512][512][3][3];
    float* d_conv4_3_weight;   // [512][512][3][3];
    float* d_conv5_1_weight;   // [512][512][3][3];
    float* d_conv5_2_weight;   // [512][512][3][3];
    float* d_conv5_3_weight;   // [512][512][3][3];

    float* d_conv1_1_bias;   // [64];
    float* d_conv1_2_bias;   // [64];
    float* d_conv2_1_bias;   // [128];
    float* d_conv2_2_bias;   // [128];
    float* d_conv3_1_bias;   // [256];
    float* d_conv3_2_bias;   // [256];
    float* d_conv3_3_bias;   // [256];
    float* d_conv4_1_bias;   // [512];
    float* d_conv4_2_bias;   // [512];
    float* d_conv4_3_bias;   // [512];
    float* d_conv5_1_bias;   // [512];
    float* d_conv5_2_bias;   // [512];
    float* d_conv5_3_bias;   // [512];

    float* d_fc1_weight;     // [512][10];
    float* d_fc1_bias;       // [10];
    //////////////////////////////////////////////////
    // Device Feature Maps
    //////////////////////////////////////////////////
    uint8_t* d_image;                   // [batch][3][32][32];
    float* d_input;                   // [batch][3][32][32];
    float* d_input_padded;             // [batch][3][34][34]

    float* d_C1_1_feature_map;        // [batch][64][32][32];
    float* d_C1_1_feature_map_padded; // [batch][64][34][34];
    float* d_C1_2_feature_map;        // [batch][64][32][32];
    float* d_S1_feature_map;          // [batch][64][16][16];
    float* d_S1_feature_map_padded;   // [batch][64][18][18];

    float* d_C2_1_feature_map;        // [batch][128][16][16];
    float* d_C2_1_feature_map_padded; // [batch][128][18][18]
    float* d_C2_2_feature_map;        // [batch][128][16][16];
    float* d_S2_feature_map;          // [batch][128][8][8];
    float* d_S2_feature_map_padded;   // [batch][128][10][10];

    float* d_C3_1_feature_map;        // [batch][256][8][8];
    float* d_C3_1_feature_map_padded; // [batch][256][10][10];
    float* d_C3_2_feature_map;        // [batch][256][8][8];
    float* d_C3_2_feature_map_padded; // [batch][256][10][10];
    float* d_C3_3_feature_map;        // [batch][256][8][8];
    float* d_S3_feature_map;          // [batch][256][4][4];
    float* d_S3_feature_map_padded;   // [batch][256][6][6];

    float* d_C4_1_feature_map;        // [batch][512][4][4];
    float* d_C4_1_feature_map_padded; // [batch][512][6][6];
    float* d_C4_2_feature_map;        // [batch][512][4][4];
    float* d_C4_2_feature_map_padded; // [batch][512][6][6];
    float* d_C4_3_feature_map;        // [batch][512][4][4];
    float* d_S4_feature_map;          // [batch][512][2][2];
    float* d_S4_feature_map_padded;   // [batch][512][4][4];

    float* d_C5_1_feature_map;        // [batch][512][2][2];
    float* d_C5_1_feature_map_padded; // [batch][512][4][4];
    float* d_C5_2_feature_map;        // [batch][512][2][2];
    float* d_C5_2_feature_map_padded; // [batch][512][4][4];
    float* d_C5_3_feature_map;        // [batch][512][2][2];
    float* d_S5_feature_map;          // [batch][512][1][1];

    float* d_output;                  // [batch][10];
    int*    d_predict_cuda;   // [batch];

     // Functions
    void cpu_normalize(const uint8_t* const image, float* input);
    void cpu_relu(float* feature_map, int size);
    void cpu_conv(float* input, float* output, float* weight, float* bias,
              int B, int H, int W, int IC, int OC, int K);
    void cpu_pool(float* input, float* output,
         int B, int C, int H, int W);
    void cpu_fc(float* input, float* output, float* weight, float* bias,
         int B, int IC, int OC);
    void cpu_softmax(float* input, int* output, int B, int size);
    // P cpu_int Funtions for debug
    void cpu_print_output(float* data) {
      for(int b = 0;b<batch;b++) {
        for (int i=0;i<output_size;i++) {
        printf("[%d][%d]: %lf\n", b,i,data[b*output_size + i]);
        }
      }
    }

 
};

#endif
