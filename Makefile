C = nvcc
NVCCFLAGS = -arch=sm_60 
CFLAGS = -std=c++11

all: cudabrain

cudabrain: cudabrain.cu  
	$(C) $(NVCCFLAGS) $(CFLAGS) -o cudabrain.exe cudabrain.cu 
clean:
	rm -f cudabrain.exe *.o *.err *.out
