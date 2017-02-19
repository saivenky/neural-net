CC=gcc
FLAGS=-O3 -fPIC
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"

all: libneuron.so
clean:
	rm libneuron.so
	rm *.o
neuron.o: neuron.c
	${CC} ${FLAGS} -c neuron.c neuron_props.c rand.c -lm -lpthread
libneuron.so: saivenky_neural_ArrayCImplConvolutionLayer.c neuron.o
	${CC} ${JNI} ${FLAGS} -o libneuron.so -shared -Wl,-soname,neuron.so saivenky_neural_ArrayCImplConvolutionLayer.c neuron.o
