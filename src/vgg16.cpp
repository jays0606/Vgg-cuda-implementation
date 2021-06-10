#include "vgg16.h"

vgg16::vgg16(int batch) {
  // Internal variable
  this->batch = batch;
  // Model Parameters
  //////////BLOCK 1/////////////////////////////////
  this->conv1_1_weight = new float[conv1_1_in_channel * conv1_1_out_channel *
                                  conv1_1_kernel_size * conv1_1_kernel_size];
  this->conv1_1_bias = new float[conv1_1_out_channel];
  this->conv1_2_weight = new float[conv1_2_in_channel * conv1_2_out_channel *
                                  conv1_2_kernel_size * conv1_2_kernel_size];
  this->conv1_2_bias = new float[conv1_2_out_channel];

  //////////BLOCK 2/////////////////////////////////
  this->conv2_1_weight = new float[conv2_1_in_channel * conv2_1_out_channel *
                                  conv2_1_kernel_size * conv2_1_kernel_size];
  this->conv2_1_bias = new float[conv2_1_out_channel];
  this->conv2_2_weight = new float[conv2_2_in_channel * conv2_2_out_channel *
                                  conv2_2_kernel_size * conv2_2_kernel_size];
  this->conv2_2_bias = new float[conv2_2_out_channel];

  //////////BLOCK 3/////////////////////////////////
  this->conv3_1_weight = new float[conv3_1_in_channel * conv3_1_out_channel *
                                  conv3_1_kernel_size * conv3_1_kernel_size];
  this->conv3_1_bias = new float[conv3_1_out_channel];
  this->conv3_2_weight = new float[conv3_2_in_channel * conv3_2_out_channel *
                                  conv3_2_kernel_size * conv3_2_kernel_size];
  this->conv3_2_bias = new float[conv3_2_out_channel];
  this->conv3_3_weight = new float[conv3_3_in_channel * conv3_3_out_channel *
                                  conv3_3_kernel_size * conv3_3_kernel_size];
  this->conv3_3_bias = new float[conv3_3_out_channel];

  //////////BLOCK 4/////////////////////////////////
  this->conv4_1_weight = new float[conv4_1_in_channel * conv4_1_out_channel *
                                  conv4_1_kernel_size * conv4_1_kernel_size];
  this->conv4_1_bias = new float[conv4_1_out_channel];
  this->conv4_2_weight = new float[conv4_2_in_channel * conv4_2_out_channel *
                                  conv4_2_kernel_size * conv4_2_kernel_size];
  this->conv4_2_bias = new float[conv4_2_out_channel];
  this->conv4_3_weight = new float[conv4_3_in_channel * conv4_3_out_channel *
                                  conv4_3_kernel_size * conv4_3_kernel_size];
  this->conv4_3_bias = new float[conv4_3_out_channel];

  //////////BLOCK 5/////////////////////////////////
  this->conv5_1_weight = new float[conv5_1_in_channel * conv5_1_out_channel *
                                  conv5_1_kernel_size * conv5_1_kernel_size];
  this->conv5_1_bias = new float[conv5_1_out_channel];
  this->conv5_2_weight = new float[conv5_2_in_channel * conv5_2_out_channel *
                                  conv5_2_kernel_size * conv5_2_kernel_size];
  this->conv5_2_bias = new float[conv5_2_out_channel];
  this->conv5_3_weight = new float[conv5_3_in_channel * conv5_3_out_channel *
                                  conv5_3_kernel_size * conv5_3_kernel_size];
  this->conv5_3_bias = new float[conv5_3_out_channel];

  //////////FC/////////////////////////////////
  this->fc1_weight = new float[fc1_in_channel * fc1_out_channel];
  this->fc1_bias = new float[fc1_out_channel];

  // Activation
  this->input = new float[batch * input_channel * input_size * input_size];
  this->input_padded = new float[batch * input_channel * (input_size+2*conv1_1_padding_size) * (input_size+2*conv1_1_padding_size)];
  this->C1_1_feature_map = new float[batch * C1_1_channel * C1_1_size * C1_1_size];
  this->C1_1_feature_map_padded = new float[batch * C1_1_channel * (C1_1_size+2*conv1_2_padding_size) * (C1_1_size+2*conv1_2_padding_size)];
  this->C1_2_feature_map = new float[batch * C1_2_channel * C1_2_size * C1_2_size];
  this->S1_feature_map = new float[batch * S1_channel * S1_size * S1_size];


  this->S1_feature_map_padded = new float[batch * S1_channel * (S1_size+2*conv2_1_padding_size) * (S1_size+2*conv2_1_padding_size)];
  this->C2_1_feature_map = new float[batch * C2_1_channel * C2_1_size * C2_1_size];
  this->C2_1_feature_map_padded = new float[batch * C2_1_channel * (C2_1_size+2*conv2_2_padding_size) * (C2_1_size+2*conv2_2_padding_size)];
  this->C2_2_feature_map = new float[batch * C2_2_channel * C2_2_size * C2_2_size];
  this->S2_feature_map = new float[batch * S2_channel * S2_size * S2_size];

  this->S2_feature_map_padded = new float[batch * S2_channel * (S2_size+2*conv3_1_padding_size) * (S2_size+2*conv3_1_padding_size)];
  this->C3_1_feature_map = new float[batch * C3_1_channel * C3_1_size * C3_1_size];
  this->C3_1_feature_map_padded = new float[batch * C3_1_channel * (C3_1_size+2*conv3_2_padding_size) * (C3_1_size+2*conv3_2_padding_size)];
  this->C3_2_feature_map = new float[batch * C3_2_channel * C3_2_size * C3_2_size];
  this->C3_2_feature_map_padded = new float[batch * C3_2_channel * (C3_2_size+2*conv3_3_padding_size) * (C3_2_size+2*conv3_3_padding_size)];
  this->C3_3_feature_map = new float[batch * C3_3_channel * C3_3_size * C3_3_size];
  this->S3_feature_map = new float[batch * S3_channel * S3_size * S3_size];

  this->S3_feature_map_padded = new float[batch * S3_channel * (S3_size+2*conv4_1_padding_size) * (S3_size+2*conv4_1_padding_size)];
  this->C4_1_feature_map = new float[batch * C4_1_channel * C4_1_size * C4_1_size];
  this->C4_1_feature_map_padded = new float[batch * C4_1_channel * (C4_1_size+2*conv4_2_padding_size) * (C4_1_size+2*conv4_2_padding_size)];
  this->C4_2_feature_map = new float[batch * C4_2_channel * C4_2_size * C4_2_size];
  this->C4_2_feature_map_padded = new float[batch * C4_2_channel * (C4_2_size+2*conv4_3_padding_size) * (C4_2_size+2*conv4_3_padding_size)];
  this->C4_3_feature_map = new float[batch * C4_3_channel * C4_3_size * C4_3_size];
  this->S4_feature_map = new float[batch * S4_channel * S4_size * S4_size];

  this->S4_feature_map_padded = new float[batch * S4_channel * (S4_size+2*conv5_1_padding_size) * (S4_size+2*conv5_1_padding_size)];
  this->C5_1_feature_map = new float[batch * C5_1_channel * C5_1_size * C5_1_size];
  this->C5_1_feature_map_padded = new float[batch * C5_1_channel * (C5_1_size+2*conv5_2_padding_size) * (C5_1_size+2*conv5_2_padding_size)];
  this->C5_2_feature_map = new float[batch * C5_2_channel * C5_2_size * C5_2_size];
  this->C5_2_feature_map_padded = new float[batch * C5_2_channel * (C5_2_size+2*conv5_3_padding_size) * (C5_2_size+2*conv5_3_padding_size)];
  this->C5_3_feature_map = new float[batch * C5_3_channel * C5_3_size * C5_3_size];
  this->S5_feature_map = new float[batch * S5_channel * S5_size * S5_size];

  this->output = new float[batch * output_size];
}

vgg16::~vgg16() {
  // Free model parameters memories
  delete[] this->conv1_1_weight;
  delete[] this->conv1_1_bias;
  delete[] this->conv1_2_weight;
  delete[] this->conv1_2_bias;

  delete[] this->conv2_1_weight;
  delete[] this->conv2_1_bias;
  delete[] this->conv2_2_weight;
  delete[] this->conv2_2_bias;

  delete[] this->conv3_1_weight;
  delete[] this->conv3_1_bias;
  delete[] this->conv3_2_weight;
  delete[] this->conv3_2_bias;
  delete[] this->conv3_3_weight;
  delete[] this->conv3_3_bias;

  delete[] this->conv4_1_weight;
  delete[] this->conv4_1_bias;
  delete[] this->conv4_2_weight;
  delete[] this->conv4_2_bias;
  delete[] this->conv4_3_weight;
  delete[] this->conv4_3_bias;

  delete[] this->conv5_1_weight;
  delete[] this->conv5_1_bias;
  delete[] this->conv5_2_weight;
  delete[] this->conv5_2_bias;
  delete[] this->conv5_3_weight;
  delete[] this->conv5_3_bias;

  delete[] this->fc1_weight;
  delete[] this->fc1_bias;

  // Free activation memories
  delete[] this->input;
  delete[] this->input_padded;
  delete[] this->C1_1_feature_map;
  delete[] this->C1_1_feature_map_padded;
  delete[] this->C1_2_feature_map;
  delete[] this->S1_feature_map;

  delete[] this->S1_feature_map_padded;
  delete[] this->C2_1_feature_map;
  delete[] this->C2_1_feature_map_padded;
  delete[] this->C2_2_feature_map;
  delete[] this->S2_feature_map;

  delete[] this->S2_feature_map_padded;
  delete[] this->C3_1_feature_map;
  delete[] this->C3_1_feature_map_padded;
  delete[] this->C3_2_feature_map;
  delete[] this->C3_2_feature_map_padded;
  delete[] this->C3_3_feature_map;
  delete[] this->S3_feature_map;

  delete[] this->S3_feature_map_padded;
  delete[] this->C4_1_feature_map;
  delete[] this->C4_1_feature_map_padded;
  delete[] this->C4_2_feature_map;
  delete[] this->C4_2_feature_map_padded;
  delete[] this->C4_3_feature_map;
  delete[] this->S4_feature_map;

  delete[] this->S4_feature_map_padded;
  delete[] this->C5_1_feature_map;
  delete[] this->C5_1_feature_map_padded;
  delete[] this->C5_2_feature_map;
  delete[] this->C5_2_feature_map_padded;
  delete[] this->C5_3_feature_map;
  delete[] this->S5_feature_map;

  delete[] this->output;
}

void vgg16::load_parameters(std::string value_path) {
  // Load parameters from value_path
  {
    // Initialize variables
    std::string buffer;
    std::ifstream value_file;
    // Open file
    value_file.open(value_path);

    //////////BLOCK 1/////////////////////////////////
    // conv1_1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv1_1_in_channel * conv1_1_out_channel; c++) {
      for (int i = 0; i < conv1_1_kernel_size; i++){
        for (int j = 0; j < conv1_1_kernel_size; j++){
          value_file >>
              conv1_1_weight[c * (conv1_1_kernel_size * conv1_1_kernel_size) +
                           i * conv1_1_kernel_size + j];
        }
      }
    }
    // conv1_2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv1_2_in_channel * conv1_2_out_channel; c++) {
      for (int i = 0; i < conv1_2_kernel_size; i++){
        for (int j = 0; j < conv1_2_kernel_size; j++){
          value_file >>
              conv1_2_weight[c * (conv1_2_kernel_size * conv1_2_kernel_size) +
                           i * conv1_2_kernel_size + j];
        }
      }
    }
    //////////BLOCK 2/////////////////////////////////
    // conv2_1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv2_1_in_channel * conv2_1_out_channel; c++) {
      for (int i = 0; i < conv2_1_kernel_size; i++)
        for (int j = 0; j < conv2_1_kernel_size; j++)
          value_file >>
              conv2_1_weight[c * (conv2_1_kernel_size * conv2_1_kernel_size) +
                           i * conv2_1_kernel_size + j];
    }
    // conv2_2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv2_2_in_channel * conv2_2_out_channel; c++) {
      for (int i = 0; i < conv2_2_kernel_size; i++)
        for (int j = 0; j < conv2_2_kernel_size; j++)
          value_file >>
              conv2_2_weight[c * (conv2_2_kernel_size * conv2_2_kernel_size) +
                           i * conv2_2_kernel_size + j];
    }
    //////////BLOCK 3/////////////////////////////////
    // conv3_1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv3_1_in_channel * conv3_1_out_channel; c++) {
      for (int i = 0; i < conv3_1_kernel_size; i++)
        for (int j = 0; j < conv3_1_kernel_size; j++)
          value_file >>
              conv3_1_weight[c * (conv3_1_kernel_size * conv3_1_kernel_size) +
                           i * conv3_1_kernel_size + j];
    }
    // conv3_2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv3_2_in_channel * conv3_2_out_channel; c++) {
      for (int i = 0; i < conv3_2_kernel_size; i++)
        for (int j = 0; j < conv3_2_kernel_size; j++)
          value_file >>
              conv3_2_weight[c * (conv3_2_kernel_size * conv3_2_kernel_size) +
                           i * conv3_2_kernel_size + j];
    }
    // conv3_3.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv3_3_in_channel * conv3_3_out_channel; c++) {
      for (int i = 0; i < conv3_3_kernel_size; i++)
        for (int j = 0; j < conv3_3_kernel_size; j++)
          value_file >>
              conv3_3_weight[c * (conv3_3_kernel_size * conv3_3_kernel_size) +
                           i * conv3_3_kernel_size + j];
    }
    //////////BLOCK 4/////////////////////////////////
    // conv4_1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv4_1_in_channel * conv4_1_out_channel; c++) {
      for (int i = 0; i < conv4_1_kernel_size; i++)
        for (int j = 0; j < conv4_1_kernel_size; j++)
          value_file >>
              conv4_1_weight[c * (conv4_1_kernel_size * conv4_1_kernel_size) +
                           i * conv4_1_kernel_size + j];
    }
    // conv4_2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv4_2_in_channel * conv4_2_out_channel; c++) {
      for (int i = 0; i < conv4_2_kernel_size; i++)
        for (int j = 0; j < conv4_2_kernel_size; j++)
          value_file >>
              conv4_2_weight[c * (conv4_2_kernel_size * conv4_2_kernel_size) +
                           i * conv4_2_kernel_size + j];
    }
    // conv4_3.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv4_3_in_channel * conv4_3_out_channel; c++) {
      for (int i = 0; i < conv4_3_kernel_size; i++)
        for (int j = 0; j < conv4_3_kernel_size; j++)
          value_file >>
              conv4_3_weight[c * (conv4_3_kernel_size * conv4_3_kernel_size) +
                           i * conv4_3_kernel_size + j];
    }
    //////////BLOCK 5/////////////////////////////////
    // conv5_1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv5_1_in_channel * conv5_1_out_channel; c++) {
      for (int i = 0; i < conv5_1_kernel_size; i++)
        for (int j = 0; j < conv5_1_kernel_size; j++)
          value_file >>
              conv5_1_weight[c * (conv5_1_kernel_size * conv5_1_kernel_size) +
                           i * conv5_1_kernel_size + j];
    }
    // conv5_2.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv5_2_in_channel * conv5_2_out_channel; c++) {
      for (int i = 0; i < conv5_2_kernel_size; i++)
        for (int j = 0; j < conv5_2_kernel_size; j++)
          value_file >>
              conv5_2_weight[c * (conv5_2_kernel_size * conv5_2_kernel_size) +
                           i * conv5_2_kernel_size + j];
    }
    // conv5_3.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int c = 0; c < conv5_3_in_channel * conv5_3_out_channel; c++) {
      for (int i = 0; i < conv5_3_kernel_size; i++){
        for (int j = 0; j < conv5_3_kernel_size; j++){
          value_file >>
              conv5_3_weight[c * (conv5_3_kernel_size * conv5_3_kernel_size) +
                           i * conv5_3_kernel_size + j];
                      
        }
      }
    }
    
    //////////BLOCK 1/////////////////////////////////
    // conv1_1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv1_1_out_channel; oc++) value_file >> conv1_1_bias[oc];

    // conv1_2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv1_2_out_channel; oc++) value_file >> conv1_2_bias[oc];

    //////////BLOCK 2/////////////////////////////////
    // conv2_1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv2_1_out_channel; oc++) value_file >> conv2_1_bias[oc];
    // conv2_2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv2_2_out_channel; oc++) value_file >> conv2_2_bias[oc];

    //////////BLOCK 3/////////////////////////////////
    // conv3_1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv3_1_out_channel; oc++) value_file >> conv3_1_bias[oc];
    // conv3_2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv3_2_out_channel; oc++) value_file >> conv3_2_bias[oc];
    // conv3_3.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv3_3_out_channel; oc++) value_file >> conv3_3_bias[oc];

    //////////BLOCK 4/////////////////////////////////
    // conv4_1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv4_1_out_channel; oc++) value_file >> conv4_1_bias[oc];
    // conv4_2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv4_2_out_channel; oc++) value_file >> conv4_2_bias[oc];
    // conv4_3.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv4_3_out_channel; oc++) value_file >> conv4_3_bias[oc];

    //////////BLOCK 5/////////////////////////////////
    // conv5_1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv5_1_out_channel; oc++) value_file >> conv5_1_bias[oc];
    // conv5_2.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv5_2_out_channel; oc++) value_file >> conv5_2_bias[oc];
    // conv5_3.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < conv5_3_out_channel; oc++) value_file >> conv5_3_bias[oc];

    // fc1.weight
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc1_out_channel; oc++) {
      for (int ic = 0; ic < fc1_in_channel; ic++) {
        value_file >> fc1_weight[oc * fc1_in_channel + ic];
      }
    }
    // fc1.bias
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    getline(value_file, buffer);
    for (int oc = 0; oc < fc1_out_channel; oc++) {
      value_file >> fc1_bias[oc];
    }

    // Close file
    value_file.close();
  }
}

void vgg16::print_parameters() {
  std::cout.precision(std::numeric_limits<float>::max_digits10);
  // conv1_1.weight
  for (int c = 0; c < conv1_1_in_channel * conv1_1_out_channel; c++) {
    std::cout << "conv1_1.weight.c" << c + 1 << std::endl;
    for (int i = 0; i < conv1_1_kernel_size; i++) {
      for (int j = 0; j < conv1_1_kernel_size; j++) {
        std::cout << conv1_1_weight[c * (conv1_1_kernel_size * conv1_1_kernel_size) +
                                  i * conv1_1_kernel_size + j]
                  << " ";
      }
      std::cout << std::endl;
    }
  }
  // conv2_1.weight
  for (int c = 0; c < conv2_1_in_channel * conv2_1_out_channel; c++) {
    std::cout << "conv2_1.weight.c" << c + 1 << std::endl;
    for (int i = 0; i < conv2_1_kernel_size; i++) {
      for (int j = 0; j < conv2_1_kernel_size; j++) {
        std::cout << conv2_1_weight[c * (conv2_1_kernel_size * conv2_1_kernel_size) +
                                  i * conv2_1_kernel_size + j]
                  << " ";
      }
      std::cout << std::endl;
    }
  }
  // conv1.bias
  std::cout << "conv1_1.bias" << std::endl;
  for (int oc = 0; oc < conv1_1_out_channel; oc++) {
    std::cout << conv1_1_bias[oc] << " ";
  }
  std::cout << std::endl;
  // conv2_1.bias
  std::cout << "conv2_1.bias" << std::endl;
  for (int oc = 0; oc < conv2_1_out_channel; oc++) {
    std::cout << conv2_1_bias[oc] << " ";
  }
  std::cout << std::endl;
  // fc1.weight
  for (int oc = 0; oc < fc1_out_channel; oc++) {
    std::cout << "fc1.weight.out_channel" << oc + 1 << std::endl;
    for (int ic = 0; ic < fc1_in_channel; ic++) {
      std::cout << fc1_weight[oc * fc1_in_channel + ic] << " ";
    }
    std::cout << std::endl;
  }
  // fc1.bias
  std::cout << "fc1.bias" << std::endl;
  for (int oc = 0; oc < fc1_out_channel; oc++) {
    std::cout << fc1_bias[oc] << " ";
  }
  std::cout << std::endl;
}

void vgg16::softmax(float* input, int* output, int B, int size) {
  for (int b = 0; b < B; b++) {
    // Initialize
    int max_idx = 0;
    float max_val = std::exp(std::numeric_limits<float>::lowest());
    // calcualte Z = sum_all(exp(x_i))
    float Z = 0;
    for (int i = 0; i < size; i++) Z += std::exp(input[b * size + i]);
    // Softmax
    for (int i = 0; i < size; i++) {
      input[b * size + i] = std::exp(input[b * size + i]) / Z;
      if (input[i] - max_val > std::numeric_limits<float>::epsilon()) {
        max_val = input[b * size + i];
        max_idx = i;
      }
    }
    output[b] = max_idx;
  }
}

bool vgg16::compare(vgg16* other) {
  // TODO: Implement this...
  return true;
}
