#include <jni.h>
#include "sigmoid_layer.h"
#include "saivenky_neural_c_SigmoidLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_create
(JNIEnv * env, jobject object, jint size, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct sigmoid_layer *layer = create_sigmoid_layer(size, inputActivation, inputError);
  SetByteBuffer(env, object, "outputSignal", layer->outputSignal, layer->size * sizeof(double));
  SetByteBuffer(env, object, "outputError", layer->outputError, layer->size * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SigmoidLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  destroy_sigmoid_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SigmoidLayer_feedforward
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {

  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  feedforward_sigmoid_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SigmoidLayer_backpropogate
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  backpropogate_sigmoid_layer(layer);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SigmoidLayer_update
(JNIEnv *env, jobject obj, jlong nativeLayerPtr, jdouble rate) {
  struct sigmoid_layer *layer = (struct sigmoid_layer *)nativeLayerPtr;
  update_sigmoid_layer(layer);
}
