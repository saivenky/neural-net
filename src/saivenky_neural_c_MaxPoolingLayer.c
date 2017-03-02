#include <jni.h>
#include "max_pooling_layer.h"
#include "jni_helper.h"
#include "network_layer.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_MaxPoolingLayer_create
(JNIEnv *env, jobject obj, jintArray inputShape, jintArray poolShape, jint stride, jobject jinputActivation, jobject jinputError) {
  struct JIntArray jinputShape, jpoolShape;
  jinputShape.jarray = inputShape;
  jpoolShape.jarray = poolShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jpoolShape);
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct max_pooling_layer *layer = create_max_pooling_layer(jinputShape.array, jpoolShape.array, stride);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_max_pooling_layer(network_layer->layer, inputActivation);
  network_layer->gradient = create_gradient_max_pooling_layer(network_layer->layer, inputError);
  copy_network_layer_buffers(env, obj, network_layer, layer->outputDim.dim2);

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
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  feedforward_max_pooling_layer(l->layer, l->activation);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_MaxPoolingLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  backpropogate_max_pooling_layer(l->layer, l->activation, l->gradient);
}
