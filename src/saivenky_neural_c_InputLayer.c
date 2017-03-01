#include <jni.h>
#include "input_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_InputLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_create
(JNIEnv *env, jobject obj, jint size) {
  struct input_layer *layer = create_input_layer(size);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_input_layer(layer);
  network_layer->gradient = create_gradient_input_layer(layer);
  copy_network_layer_buffers(env, obj, network_layer, size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct input_layer *layer = (struct input_layer *)nativeLayerPtr;
  destroy_input_layer(layer);
}
