#include <jni.h>
#include "relu_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_ReluLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_create
(JNIEnv * env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct relu_layer *layer = create_relu_layer(size);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_relu_layer,
      (create_gradient_type)&create_gradient_relu_layer);
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
  feedforward_network_layer((void *)nativeLayerPtr, (feedforward_type)&feedforward_relu_layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  backpropogate_network_layer((void *)nativeLayerPtr, (backpropogate_type)&backpropogate_relu_layer);
}
