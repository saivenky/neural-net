#include <jni.h>
#include "network_layer.h"
#include "fully_connected_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_create
(JNIEnv *env, jobject obj, jlong inputSize, jlong outputSize, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct fully_connected_layer *layer = create_fully_connected_layer(inputSize, outputSize);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_fully_connected_layer,
      (create_gradient_type)&create_gradient_fully_connected_layer,
      (feedforward_type)&feedforward_fully_connected_layer,
      (backpropogate_type)&backpropogate_fully_connected_layer,
      (update_type)&update_fully_connected_layer);

  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_FullyConnectedLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct fully_connected_layer *l = (struct fully_connected_layer *)nativeLayerPtr;
  destroy_fully_connected_layer(l);
}
