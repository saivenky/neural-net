#include "saivenky_neural_c_Layer.h"
#include "network_layer.h"

JNIEXPORT void JNICALL Java_saivenky_neural_c_Layer_feedforward
(JNIEnv *env, jclass clazz, jlong nativeLayerPtr) {
  feedforward_network_layer((void *)nativeLayerPtr);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_Layer_backpropogate
(JNIEnv *env, jclass clazz, jlong nativeLayerPtr) {
  backpropogate_network_layer((void *)nativeLayerPtr);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_Layer_update
(JNIEnv *env, jclass clazz, jlong nativeLayerPtr, jdouble rate) {
  update_network_layer((void *)nativeLayerPtr, rate);
}
