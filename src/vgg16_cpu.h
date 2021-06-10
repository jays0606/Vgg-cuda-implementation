#ifndef VGG16_CPU_H
#define VGG16_CPU_H

#include "vgg16.h"

class vgg16_cpu : public vgg16
{
public:
    // Get from base class
    void load_parameters(std::string value_path) override { vgg16::load_parameters(value_path); };
    void print_parameters() override { vgg16::print_parameters(); };
    bool compare(vgg16* other) override { return vgg16::compare(other); };
    // Implement!
    vgg16_cpu(int batch = 1) : vgg16(batch) {};
    ~vgg16_cpu() {};
    void predict(const uint8_t* const image, int batch) override;
    void classify(int* predict, int batch) override;
private:
    // Functions
    void normalize(const uint8_t* const image, float* input);
    void relu(float* feature_map, int size);
    void pad(float* input, float* input_padded,
             int B, int C, int H, int W, int P);
    void conv(float* input, float* output, float* weight, float* bias,
              int B, int H, int W, int IC, int OC, int K);
    void pool(float* input, float* output,
              int B, int C, int H, int W);
    void fc(float* input, float* output, float* weight, float* bias,
            int B, int IC, int OC);
    // Print Funtions for debug
    void print_fc(float* data, int size);
    void print_C1();
    void print_C3();
};

#endif
