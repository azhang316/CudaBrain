C = nvcc
NVCCFLAGS = -arch=sm_60 
CFLAGS = -std=c++11

all: cudabrain

cudabrain: cudabrain.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o cudabrain.exe cudabrain.cu 
matmulc: matmul.c  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o matmul.exe matmul.c
clean:
	rm -f cudabrain.exe *.o *.err *.out
