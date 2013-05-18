all:
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept.c gdaemon.c -o gdaemon.so
async: 
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_async.c -o ic_async.so
stream: 
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_stream.c -o ic_stream.so
run:
	LD_PRELOAD=`pwd`/gdaemon.so ./test
run_async:
	LD_PRELOAD=`pwd`/ic_async.so ./test
run_stream:
	LD_PRELOAD=`pwd`/ic_stream.so ./test
clean:
	rm -f test gdaemon.so ic_async.so ic_stream.so
