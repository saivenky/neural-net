CC=gcc
FLAGS=-O3 -fPIC -Wfloat-conversion
JNI=-I"/usr/lib/jvm/java-8-oracle/include" -I"/usr/lib/jvm/java-8-oracle/include/linux"

all: libneural.so
clean:
	rm *.so *.o
activation.o: activation.c
	${CC} ${FLAGS} -c activation.c
gradient.o: gradient.c
	${CC} ${FLAGS} -c gradient.c
jni_helper.o: jni_helper.c
	${CC} ${JNI} ${FLAGS} -c jni_helper.c
rand.o: rand.c
	${CC} ${FLAGS} -c rand.c -lm
kernel_dim.o: kernel_dim.c
	${CC} ${FLAGS} -c kernel_dim.c
sigmoid_layer.o: sigmoid_layer.c
	${CC} ${FLAGS} -c sigmoid_layer.c -lm
neural_network.o: neural_network.c
	${CC} ${FLAGS} -c neural_network.c -pthread
%_layer.o: %_layer.c
	${CC} ${FLAGS} -c $<
libneural.so: saivenky_neural_c_*.c $(patsubst %.c,%.o,$(wildcard *_layer.c)) jni_helper.o rand.o kernel_dim.o activation.o gradient.o neural_network.o
	${CC} ${JNI} ${FLAGS} -o libneural.so -shared -Wl,-soname,neural.so $^
