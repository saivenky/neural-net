#include <jni.h>
#include "sigmoid_layer.h"
#include "network_layer.h"
#include "saivenky_neural_c_SigmoidLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_create
(JNIEnv * env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct sigmoid_layer *layer = create_sigmoid_layer(size);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_sigmoid_layer,
      (create_gradient_type)&create_gradient_sigmoid_layer,
      (feedforward_type)&feedforward_sigmoid_layer,
      (backpropogate_type)&backpropogate_sigmoid_layer,
      NULL);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  destroy_sigmoid_layer(layer);
}
