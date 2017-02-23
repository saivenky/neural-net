#include <jni.h>
#include "fully_connected_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_create
(JNIEnv *env, jobject obj, jlong inputSize, jlong outputSize, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct fully_connected_layer *layer = create_fully_connected_layer(inputSize, outputSize, inputActivation, inputError);
  SetByteBuffer(env, obj, "outputSignal", layer->outputSignal, outputSize * sizeof(double));
  SetByteBuffer(env, obj, "outputError", layer->outputError, outputSize * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  destroy_fully_connected_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  feedforward_fully_connected_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  backpropogate_fully_connected_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_update
(JNIEnv *env, jobject obj, jlong nativeLayerPtr, jdouble rate) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  update_fully_connected_layer(l, rate);
}
