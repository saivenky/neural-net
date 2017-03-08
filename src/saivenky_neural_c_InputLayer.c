#include <jni.h>
#include "input_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_InputLayer.h"
#include "jni_helper.h"

struct activation create_activation_input_layer_wrapper(void *layer, float_t *inputActivation) {
  return create_activation_input_layer(layer);
}

struct gradient create_gradient_input_layer_wrapper(void *layer, float_t *inputError) {
  return create_gradient_input_layer(layer);
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_create
(JNIEnv *env, jobject obj, jint size, jint miniBatchSize) {
  struct input_layer *layer = create_input_layer(size);
  struct network_layer param_layer;
  param_layer.miniBatchSize = miniBatchSize;
  param_layer.activations = NULL;
  param_layer.gradients = NULL;
  struct network_layer *network_layer = create_network_layer(
      layer,
      &param_layer,
      (create_activation_type)&create_activation_input_layer_wrapper,
      (create_gradient_type)&create_gradient_input_layer_wrapper,
      NULL,
      NULL,
      NULL);
  copy_network_layer_buffers(env, obj, network_layer, size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct input_layer *layer = (struct input_layer *)nativeLayerPtr;
  destroy_input_layer(layer);
}
