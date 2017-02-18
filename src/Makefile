CC=g++
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"
all: libneuron.so
clean:
	rm libneuron.so
	rm *.o
neuron.o: neuron.c
	gcc -O3 -fPIC -c neuron.c -lm -lpthread
libneuron.so: saivenky_neural_ArrayCImplConvolutionLayer.c neuron.o
	gcc ${JNI} -O3 -fPIC -o libneuron.so -shared -Wl,-soname,neuron.so saivenky_neural_ArrayCImplConvolutionLayer.c neuron.o
