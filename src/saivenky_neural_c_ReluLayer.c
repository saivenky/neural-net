#include <jni.h>
#include "relu_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_ReluLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_create
(JNIEnv * env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  double *inputActivation = previousLayer->activation.outputSignal;
  double *inputError = previousLayer->gradient.outputError;
  struct relu_layer *layer = create_relu_layer(size);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_relu_layer(layer, inputActivation);
  network_layer->gradient = create_gradient_relu_layer(layer, inputError);
  copy_network_layer_buffers(env, obj, network_layer, layer->size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct relu_layer *layer = (struct relu_layer *)nativeLayerPtr;
  destroy_relu_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  feedforward_relu_layer(l->layer, l->activation);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  backpropogate_relu_layer(l->layer, l->activation, l->gradient);
}
