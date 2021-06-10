#include "vgg16_cpu.h"
#include "vgg16_cuda.h"
#include "common.h"
#include "util.h"

/*
Summary:
    Return format string
*/
template <typename... Args>
std::string format_string(const std::string& format, Args... args) {
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1;
  std::unique_ptr<char[]> buffer(new char[size]);
  snprintf(buffer.get(), size, format.c_str(), args...);
  return std::string(buffer.get(), buffer.get() + size - 1);
}

int main(int argc, char** argv) {
  // Initialize arguments
  std::string input_path = "/nfs/home/mgp2021_data/cifar10/test_batch.bin";
  int data_offset = 0;
  int batch = 1;
  std::string img_path_template = "tmp/cifar10_test_%d_%s.bmp";
  std::string parameter_path = "model/values.txt";
  // Read arguments
  if (argc == 1) {
    std::cout << "[INFO] Use default arguments" << std::endl;
  } else if (argc == 6) {
    input_path = argv[1];
    data_offset = atoi(argv[2]);
    batch = atoi(argv[3]);
    img_path_template = argv[4];
    parameter_path = argv[5];
  } else {
    std::cout << "[ERROR] Invalid arguments" << std::endl;
    std::cout
        << "Usage: ./predict INPUT_PATH DATA_OFFSET BATCH IMG_PATH_TEMPLATE"
        << std::endl;
    std::cout << "    INPUT_PATH: path to input data, e.g. "
                 "/nfs/home/mgp2021_data/cifar10/test_batch.bin"
              << std::endl;
    std::cout << "    DATA_OFFSET: data_offset for input data, e.g. 0"
              << std::endl;
    std::cout << "    BATCH: batch size to inference, e.g. 1" << std::endl;
    std::cout
        << "    IMG_PATH_TEMPLATE: path template to img, %d will data_offset "
           "and %s will be label, e.g. tmp/cifar10_test_%d_%s.bmp"
        << std::endl;
    std::cout << "    PARAMETER_PATH: path to parameter, e.g. model/values.txt"
              << std::endl;
    exit(-1);
  }
  // Show arguments
  std::cout << "[INFO] Arguments will be as following: " << std::endl;
  std::cout << "    INPUT_PATH: " << input_path << std::endl;
  std::cout << "    DATA_OFFSET: " << data_offset << std::endl;
  std::cout << "    BATCH: " << batch << std::endl;
  std::cout << "    IMG_PATH_TEMPLATE: " << img_path_template << std::endl;
  std::cout << "    PARAMETER_PATH: " << parameter_path << std::endl;
  // Initialize variables
  std::cout << "[INFO] Initialize variables" << std::endl;
  auto label_dict = get_label_dict();
  uint8_t* image;
  int label[batch];
  // Allocate memories
  std::cout << "[INFO] Allocate memories" << std::endl;
  image = new uint8_t[batch * IMG_SIZE];
  // Read image
  std::cout << "[INFO] Read image from data_offset " << data_offset << " at "
            << input_path << std::endl;
  if (!read_image(input_path, data_offset, batch, image, label)) {
    std::cout << "[ERROR] Failed to read image" << std::endl;
    exit(-1);
  }
  // Save image
  for (int b = 0; b < batch; b++) {
    std::string img_path =
        format_string(img_path_template, data_offset + b,
                      label_dict.find(label[b])->second.c_str());
    std::cout << "[INFO] Save image to " << img_path << std::endl;
    save_image(img_path, image + b * (IMG_CHANNEL * IMG_HEIGHT * IMG_WIDTH));
  }

  // Predict image with CPU
 
    // hard-coded result of code below.
  int predict_cpu[batch] = {3,8,8,0,6,9,1,2,3,1,0,9,5,7,9,8,5,7,8,6,9,9,4,9,4,2,9,0,9,6,6,5,9,3,9,9,4,1,9,5,4,6,5,6,0,9,3,9,7,9,9,8,9,3,8,8,7,3,3,3,7,3,9,3,6,9,1,2,3,9,9,6,8,8,0,2,9,3,3,8,8,1,1,7,2,9,9,9,8,9,0,9,8,6,4,3,6,0,0,7,4,5,6,3,1,1,9,6,8,7,9,0,2,2,1,3,0,4,2,7,8,3,1,2,8,8,8,3};

  // int predict_cpu[batch]; // if you want to test the code below, uncomment this line and comment hard-coded result.
  // vgg16_cpu* net_cpu = new vgg16_cpu(batch);
  // {
  //   // Start cuda timer
  //   cudaEvent_t start, stop;
  //   float cudaElapsedTime;
  //   cudaEventCreate(&start);
  //   cudaEventCreate(&stop);
  //   cudaEventRecord(start, 0);
  //   // Predict image with CPU
  //   net_cpu->load_parameters(parameter_path);
  //   net_cpu->predict(image, batch); 
  //   // Stop cuda timer
  //   cudaEventRecord(stop, 0);
  //   cudaEventSynchronize(stop);
  //   cudaEventElapsedTime(&cudaElapsedTime, start, stop);
  //   std::cout << "[INFO] CPU  elapsed time is " << cudaElapsedTime << " msec"
  //             << std::endl;
  // }
  // net_cpu->classify(predict_cpu, batch);  // softmax left out from time checks
  // delete net_cpu;
  
  // Predict with CUDA
  int predict_cuda[batch];
  vgg16_cuda* net_cuda = new vgg16_cuda(batch);
  net_cuda->load_parameters(parameter_path);
  net_cuda->prepare_device_memory(image);  // load parameters into device memory
  {
    // Start cuda timer
    cudaEvent_t start, stop;
    float cudaElapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Predict image with CUDA
    net_cuda->predict(batch);
    // Stop cuda timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    std::cout << "[INFO] CUDA elapsed time is " << cudaElapsedTime << " msec"
              << std::endl;
  }

  net_cuda->classify(predict_cuda,
                     batch);  // softmax is left out from execution time
  // Print predicted class with CPU, CUDA and label
  std::cout << "[INFO] CUDA predict is as following:" << std::endl;
  printf("CPU:CLASS(NUMBER,T/F),CUDA:CLASS(NUMBER,T/F),Label:CLASS(NUMBER)\n");
  for (int b = 0; b < batch; b++) {
    printf("CPU: %10s(%d,%d), CUDA: %10s(%d,%d), Label: %10s(%d)\n",
           label_dict[predict_cpu[b]].c_str(), predict_cpu[b],
           predict_cpu[b] == label[b], label_dict[predict_cuda[b]].c_str(),
           predict_cuda[b], predict_cuda[b] == label[b],
           label_dict[label[b]].c_str(), label[b]);
  }
  // Check sanity naively
  bool sanity = true;
  int cpu_error = 0;
  int cuda_error = 0;
  for (int b = 0; b < batch; b++) {
    if (predict_cpu[b] != predict_cuda[b]) {
      sanity = false;
    }
    if (predict_cpu[b] != label[b]) {
      cpu_error++;
    }
    if (predict_cuda[b] != label[b]) {
      cuda_error++;
    }
  }
  // Check sanity: We will use more precise compare when grading
  // bool sanity = net_cpu->compare(net_cuda);
  // Print the result of sanity check
  if (std::abs(cpu_error - cuda_error) <
      0.05 * batch) {  // allow 5% accuracy drop
    std::cout << std::endl << "Correct" << std::endl << std::endl;
  } else {
    std::cout << std::endl << "!!!!!Incorrect!!!!!" << std::endl << std::endl;
  }
  std::cout << "CPU error:" << cpu_error * 100.0 / batch
            << "\% GPU error:" << cuda_error * 100.0 / batch << "\%"
            << std::endl;

  // Free memories
  delete[] image;
  delete net_cuda;
  return 0;
}
