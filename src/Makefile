CC=gcc
FLAGS=-O3 -fPIC
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"

all: libneuron.so
clean:
	rm libneuron.so
	rm *.o
rand.o: rand.c
	${CC} ${FLAGS} -c rand.c -lm
neuron_props.o: neuron_props.c rand.o
	${CC} ${FLAGS} -c neuron_props.c
neuron.o: neuron.c neuron_props.o
	${CC} ${FLAGS} -c neuron.c -lpthread
libneuron.so: saivenky_neural_ConvolutionCImplLayer.c neuron.o
	${CC} ${JNI} ${FLAGS} -o libneuron.so -shared -Wl,-soname,neuron.so saivenky_neural_ConvolutionCImplLayer.c neuron.o neuron_props.o rand.o
