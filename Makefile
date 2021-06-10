#########################
# Variables
#########################
NVCC=/usr/local/cuda-10.2/bin/nvcc
BUILD=build
SRC=src
FLAGS=-std=c++11 -O3 $(shell pkg-config --cflags opencv4)
LIBS=$(shell pkg-config --libs opencv4)

#########################
# Main
#########################
predict: $(BUILD)/main.o $(BUILD)/util.o $(BUILD)/vgg16_cpu.o $(BUILD)/vgg16_cuda.o $(BUILD)/vgg16.o 
	$(NVCC) $(FLAGS) $(LIBS) -o $@ $^

$(BUILD)/main.o: $(SRC)/main.cpp $(BUILD)/util.o $(BUILD)/vgg16.o $(BUILD)/vgg16_cpu.o $(BUILD)/vgg16_cuda.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/util.o: $(SRC)/util.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $< 

$(BUILD)/vgg16_cpu.o: $(SRC)/vgg16_cpu.cpp $(BUILD)/vgg16.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/vgg16_cuda.o: $(SRC)/vgg16_cuda.cu $(BUILD)/vgg16.o
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

$(BUILD)/vgg16.o: $(SRC)/vgg16.cpp
	$(NVCC) $(FLAGS) $(LIBS) -o $@ -c $<

run_on_server: predict
	mkdir -p result
	condor_submit predict_b128.cmd

#########################
# Util
#########################
format:
	clang-format -i -style=Google $(SRC)/*.cu $(SRC)/*.cpp

clean:
	rm -rf predict result/* tmp/*.log tmp/*.bmp $(BUILD)/*.o

queue:
	condor_q
