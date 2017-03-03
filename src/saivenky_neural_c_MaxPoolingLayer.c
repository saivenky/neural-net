#include <jni.h>
#include "max_pooling_layer.h"
#include "jni_helper.h"
#include "network_layer.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_MaxPoolingLayer_create
(JNIEnv *env, jobject obj, jintArray inputShape, jintArray poolShape, jint stride, jlong previousLayerNativePtr) {
  struct JIntArray jinputShape, jpoolShape;
  jinputShape.jarray = inputShape;
  jpoolShape.jarray = poolShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jpoolShape);
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct max_pooling_layer *layer = create_max_pooling_layer(jinputShape.array, jpoolShape.array, stride);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_max_pooling_layer,
      (create_gradient_type)&create_gradient_max_pooling_layer);

  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jpoolShape, JNI_ABORT);
  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_MaxPoolingLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct max_pooling_layer *l = (struct max_pooling_layer *)nativeLayerPtr;
  destroy_max_pooling_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_MaxPoolingLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  feedforward_network_layer((void *)nativeLayerPtr, (feedforward_type)&feedforward_max_pooling_layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_MaxPoolingLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  backpropogate_network_layer((void *)nativeLayerPtr, (backpropogate_type)&backpropogate_max_pooling_layer);
}
