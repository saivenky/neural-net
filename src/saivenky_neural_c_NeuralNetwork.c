#include <jni.h>
#include "saivenky_neural_c_NeuralNetwork.h"
#include "neural_network.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_NeuralNetwork_create
(JNIEnv *env, jclass clazz, jlongArray nativeLayerPtrs) {
  struct JArray jnativeLayerPtrs = GetLongArray(env, nativeLayerPtrs);
  struct neural_network *nn = create_neural_network(jnativeLayerPtrs.array, (*env)->GetArrayLength(env, nativeLayerPtrs));
  ReleaseArray(&jnativeLayerPtrs, JNI_ABORT);
  return (jlong)nn;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_run
(JNIEnv *env, jclass clazz, jlong nativePtr) {
  struct neural_network *nn = (struct neural_network *)nativePtr;
  feedforward_neural_network(nn);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_update
(JNIEnv *env, jclass clazz, jlong nativePtr, jfloat rate) {
  struct neural_network *nn = (struct neural_network *)nativePtr;
  update_neural_network(nn, rate);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_train
(JNIEnv *env, jclass clazz, jlong nativePtr) {
  struct neural_network *nn = (struct neural_network *)nativePtr;
  feedforward_neural_network(nn);
  backpropogate_neural_network(nn);
}
