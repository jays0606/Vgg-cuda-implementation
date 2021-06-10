############################################
##
## CUDA_VGG Condor command file
##
############################################

executable	 = predict
output		 = result/vgg16_b128.out
error		 = result/vgg16_b128.err
log		     = result/vgg16_b128.log
request_cpus = 1
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = /nfs/home/mgp2021_data/cifar10/test_batch.bin
transfer_output_files   = tmp
arguments	              = test_batch.bin 0 128 tmp/cifar10_test_%d_%s.bmp /nfs/home/mgp2021_data/vgg_weight/values_vgg.txt
queue
