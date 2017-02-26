#include <jni.h>
#include "neuron_props.h"
#include "softmax_cross_entropy_layer.h"
#include "saivenky_neural_c_SoftmaxCrossEntropyLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_create
(JNIEnv *env, jobject obj, jint size, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct softmax_cross_entropy_layer *layer = create_softmax_cross_entropy_layer(size, inputActivation, inputError);
  SetByteBuffer(env, obj, "inputActivation", layer->inputActivation, size * sizeof(double));
  SetByteBuffer(env, obj, "inputError", layer->inputError, size * sizeof(double));
  SetByteBuffer(env, obj, "outputSignal", layer->outputSignal, size * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_feedforward
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct softmax_cross_entropy_layer *layer = (struct softmax_cross_entropy_layer *)nativeLayerPtr;
    feedforward_softmax_cross_entropy_layer(layer);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_setExpected
  (JNIEnv *env, jobject object, jlong nativeLayerPtr, jdoubleArray expected) {
    struct softmax_cross_entropy_layer *layer = (struct softmax_cross_entropy_layer *)nativeLayerPtr;
    struct JDoubleArray jexpected;
    jexpected.jarray = expected;
    GetDoubleArray(env, &jexpected);
    set_expected_softmax_cross_entropy_layer(layer, jexpected.array);
    ReleaseDoubleArray(env, &jexpected, JNI_ABORT);
  }
