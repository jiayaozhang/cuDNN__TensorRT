#g++ -O3 -c cpu.cpp
#g++ -O3 -c main.cpp
#g++ -O3 -o VectorSum main.o  cpu.o
#g++ -O3 main_cpu.cpp -o VectorSumCPU

C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin matrixMultiple.cu -o GPU

