#include <jni.h>
#include "relu_layer.h"
#include "saivenky_neural_c_ReluLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_create
(JNIEnv * env, jobject object, jint size, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct relu_layer *layer = create_relu_layer(size, inputActivation, inputError);
  SetByteBuffer(env, object, "outputSignal", layer->outputSignal, layer->size * sizeof(double));
  SetByteBuffer(env, object, "outputError", layer->outputError, layer->size * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct relu_layer *layer = (struct relu_layer *)nativeLayerPtr;
  destroy_relu_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {

  struct relu_layer *layer = (struct relu_layer *)nativeLayerPtr;
  feedforward_relu_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct relu_layer *layer = (struct relu_layer *)nativeLayerPtr;
  backpropogate_relu_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_update
(JNIEnv *env, jobject obj, jlong nativeLayerPtr, jdouble rate) {
  struct relu_layer *layer = (struct relu_layer *)nativeLayerPtr;
  update_relu_layer(layer);
}
