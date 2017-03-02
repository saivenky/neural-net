#include <jni.h>
#include "saivenky_neural_c_OutputLayer.h"
#include "network_layer.h"
#include "output_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_create
(JNIEnv *env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  double *inputActivation = previousLayer->activation.outputSignal;
  double *inputError = previousLayer->gradient.outputError;
  struct output_layer *layer = create_output_layer(size);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_output_layer(inputActivation);
  network_layer->gradient = create_gradient_output_layer(inputError);
  copy_network_layer_buffers(env, obj, network_layer, size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct output_layer *layer = (struct output_layer *)nativeLayerPtr;
  destroy_output_layer(layer);
}
