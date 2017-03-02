#include <jni.h>
#include "network_layer.h"
#include "fully_connected_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_create
(JNIEnv *env, jobject obj, jlong inputSize, jlong outputSize, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  double *inputActivation = previousLayer->activation.outputSignal;
  double *inputError = previousLayer->gradient.outputError;

  struct fully_connected_layer *layer = create_fully_connected_layer(inputSize, outputSize);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_fully_connected_layer(layer, inputActivation);
  network_layer->gradient = create_gradient_fully_connected_layer(layer, inputError);
  copy_network_layer_buffers(env, obj, network_layer, outputSize);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  destroy_fully_connected_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  feedforward_fully_connected_layer(l->layer, l->activation);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  backpropogate_fully_connected_layer(l->layer, l->activation, l->gradient);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_update
(JNIEnv *env, jobject obj, jlong nativeLayerPtr, jdouble rate) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  update_fully_connected_layer(l->layer, rate);
}
