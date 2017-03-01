#include <jni.h>
#include "neuron_props.h"
#include "network_layer.h"
#include "softmax_cross_entropy_layer.h"
#include "saivenky_neural_c_SoftmaxCrossEntropyLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_create
(JNIEnv *env, jobject obj, jint size, jobject jinputActivation, jobject jinputError) {
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);

  struct softmax_cross_entropy_layer *layer = create_softmax_cross_entropy_layer(size);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_softmax_cross_entropy_layer(network_layer->layer, inputActivation);
  network_layer->gradient = create_gradient_softmax_cross_entropy_layer(inputError);
  copy_network_layer_buffers(env, obj, network_layer, size);

  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_feedforward
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct network_layer *l = (struct network_layer *)nativeLayerPtr;
    feedforward_softmax_cross_entropy_layer(l->layer, l->activation);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_setExpected
  (JNIEnv *env, jobject object, jlong nativeLayerPtr, jdoubleArray expected) {
    struct network_layer *l = (struct network_layer *)nativeLayerPtr;
    struct JDoubleArray jexpected;
    jexpected.jarray = expected;
    GetDoubleArray(env, &jexpected);
    set_expected_softmax_cross_entropy_layer(l->layer, l->activation, l->gradient, jexpected.array);
    ReleaseDoubleArray(env, &jexpected, JNI_ABORT);
  }
