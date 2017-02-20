CC=gcc
FLAGS=-O3 -fPIC
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"

all: libneural.so
clean:
	rm *.so *.o
rand.o: rand.c
	${CC} ${FLAGS} -c rand.c -lm
neuron_props.o: neuron_props.c rand.o
	${CC} ${FLAGS} -c neuron_props.c
convolution_layer.o: convolution_layer.c neuron_props.o
	${CC} ${FLAGS} -c convolution_layer.c -lpthread
libneural.so: saivenky_neural_ConvolutionCImplLayer.c convolution_layer.o
	${CC} ${JNI} ${FLAGS} -o libneural.so -shared -Wl,-soname,neural.so saivenky_neural_ConvolutionCImplLayer.c convolution_layer.o neuron_props.o rand.o
