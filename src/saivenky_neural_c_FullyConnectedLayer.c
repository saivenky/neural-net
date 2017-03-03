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
      (create_gradient_type)&create_gradient_fully_connected_layer);

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
  feedforward_network_layer((void *)nativeLayerPtr, (feedforward_type)&feedforward_fully_connected_layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  backpropogate_network_layer((void *)nativeLayerPtr, (backpropogate_type)&backpropogate_fully_connected_layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_FullyConnectedLayer_update
(JNIEnv *env, jobject obj, jlong nativeLayerPtr, jdouble rate) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  for (int i = 0; i < l->miniBatchSize; i++) {
    update_fully_connected_layer(l->layer, rate, l->gradients[i]);
  }
}
