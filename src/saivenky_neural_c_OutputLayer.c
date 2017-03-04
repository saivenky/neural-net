#include <jni.h>
#include "saivenky_neural_c_OutputLayer.h"
#include "network_layer.h"
#include "output_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_create
(JNIEnv *env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct output_layer *layer = create_output_layer(size);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_output_layer,
      (create_gradient_type)&create_gradient_output_layer,
      NULL,
      NULL,
      NULL);
  copy_network_layer_buffers(env, obj, network_layer, size);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct output_layer *layer = (struct output_layer *)nativeLayerPtr;
  destroy_output_layer(layer);
}
