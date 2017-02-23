#include <jni.h>
#include "input_layer.h"
#include "saivenky_neural_c_InputLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_create
(JNIEnv *env, jobject obj, jint size) {
  struct input_layer *layer = create_input_layer(size);
  SetByteBuffer(env, obj, "outputSignal", layer->outputSignal, size * sizeof(double));
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_InputLayer_destroy
(JNIEnv *env, jobject obj, jlong nativeLayerPtr) {
  struct input_layer *layer = (struct input_layer *)nativeLayerPtr;
  destroy_input_layer(layer);
}
