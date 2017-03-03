#include <jni.h>
#include "network_layer.h"
#include "softmax_cross_entropy_layer.h"
#include "saivenky_neural_c_SoftmaxCrossEntropyLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_create
(JNIEnv *env, jobject obj, jint size, jlong previousLayerNativePtr) {
  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct softmax_cross_entropy_layer *layer = create_softmax_cross_entropy_layer(size);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_softmax_cross_entropy_layer,
      (create_gradient_type)&create_gradient_softmax_cross_entropy_layer,
      (feedforward_type)&feedforward_softmax_cross_entropy_layer,
      NULL,
      NULL);
  copy_network_layer_buffers(env, obj, network_layer, size);

  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_SoftmaxCrossEntropyLayer_setExpected
(JNIEnv *env, jobject object, jlong nativeLayerPtr, jobjectArray expected) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  for (int i = 0; i < l->miniBatchSize; i++) {
    struct JDoubleArray jexpected;
    jexpected.jarray = (*env)->GetObjectArrayElement(env, expected, i);
    GetDoubleArray(env, &jexpected);
    set_expected_softmax_cross_entropy_layer(l->layer, l->activations[i], l->gradients[i], jexpected.array);
    ReleaseDoubleArray(env, &jexpected, JNI_ABORT);
  }
}
