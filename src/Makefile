CC=gcc
FLAGS=-O3 -fPIC
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"

all: libneural.so
clean:
	rm *.so *.o
jni_helper.o: jni_helper.c
	${CC} ${JNI} ${FLAGS} -c jni_helper.c
rand.o: rand.c
	${CC} ${FLAGS} -c rand.c -lm
neuron_props.o: neuron_props.c rand.o
	${CC} ${FLAGS} -c neuron_props.c
sigmoid_layer.o: sigmoid_layer.c
	${CC} ${FLAGS} -c sigmoid_layer.c -lm
convolution_layer.o: convolution_layer.c neuron_props.o
	${CC} ${FLAGS} -c convolution_layer.c
libneural.so: saivenky_neural_c_ConvolutionLayer.c convolution_layer.o sigmoid_layer.o jni_helper.o
	${CC} ${JNI} ${FLAGS} -o libneural.so -shared -Wl,-soname,neural.so saivenky_neural_c_*.c convolution_layer.o neuron_props.o rand.o sigmoid_layer.o jni_helper.o
