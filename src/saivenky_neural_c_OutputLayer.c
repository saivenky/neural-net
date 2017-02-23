#include <jni.h>
#include "saivenky_neural_c_OutputLayer.h"
#include "output_layer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_create
(JNIEnv *env, jobject obj, jint size, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct output_layer *layer = create_output_layer(size, inputActivation, inputError);
  SetByteBuffer(env, obj, "inputActivation", layer->inputActivation, size * sizeof(double));
  SetByteBuffer(env, obj, "inputError", layer->inputError, size * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_OutputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct output_layer *layer = (struct output_layer *)nativeLayerPtr;
  destroy_output_layer(layer);
}
