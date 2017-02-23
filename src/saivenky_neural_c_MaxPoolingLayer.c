#include <jni.h>
#include "max_pooling_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_MaxPoolingLayer_create
(JNIEnv *env, jobject obj, jintArray inputShape, jintArray poolShape, jint stride, jobject jinputActivation, jobject jinputError) {
  struct JIntArray jinputShape, jpoolShape;
  jinputShape.jarray = inputShape;
  jpoolShape.jarray = poolShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jpoolShape);
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct max_pooling_layer *layer = create_max_pooling_layer(jinputShape.array, jpoolShape.array, stride, inputActivation, inputError);
  SetByteBuffer(env, obj, "outputSignal", layer->outputSignal, layer->outputDim.dim2 * sizeof(double));
  SetByteBuffer(env, obj, "outputError", layer->outputError, layer->outputDim.dim2 * sizeof(double));
  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jpoolShape, JNI_ABORT);
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_MaxPoolingLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct max_pooling_layer *l = (struct max_pooling_layer *)nativeLayerPtr;
  destroy_max_pooling_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_MaxPoolingLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct max_pooling_layer *l = (struct max_pooling_layer *)nativeLayerPtr;
  feedforward_max_pooling_layer(l);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_MaxPoolingLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct max_pooling_layer *l = (struct max_pooling_layer *)nativeLayerPtr;
  backpropogate_max_pooling_layer(l);
}
