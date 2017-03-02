#include <jni.h>
#include "sigmoid_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_SigmoidLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_create
(JNIEnv * env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  double *inputActivation = previousLayer->activation.outputSignal;
  double *inputError = previousLayer->gradient.outputError;
  struct sigmoid_layer *layer = create_sigmoid_layer(size);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_sigmoid_layer(layer, inputActivation);
  network_layer->gradient = create_gradient_sigmoid_layer(layer, inputError);
  copy_network_layer_buffers(env, obj, network_layer, layer->size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  destroy_sigmoid_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SigmoidLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  feedforward_sigmoid_layer(l->layer, l->activation);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SigmoidLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  backpropogate_sigmoid_layer(l->layer, l->activation, l->gradient);
}
