#include <jni.h>
#include "network_layer.h"
#include "convolution_layer.h"
#include "saivenky_neural_c_ConvolutionLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ConvolutionLayer_create
(JNIEnv * env, jobject obj, jintArray inputShape, jintArray kernelShape, jint frames, jint stride, jint padding,
 jlong previousLayerNativePtr) {
  struct JIntArray jinputShape, jkernelShape;
  jinputShape.jarray = inputShape;
  jkernelShape.jarray = kernelShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jkernelShape);

  struct network_layer *previousLayer = (struct network_layer *) previousLayerNativePtr;
  struct convolution_layer *layer = create_convolution_layer(jinputShape.array, jkernelShape.array, frames, stride, padding);
  struct network_layer *network_layer = create_network_layer(
      layer,
      previousLayer,
      (create_activation_type)&create_activation_convolution_layer,
      (create_gradient_type)&create_gradient_convolution_layer,
      (feedforward_type)&feedforward_convolution_layer,
      (backpropogate_type)&backpropogate_convolution_layer,
      (update_type)&update_convolution_layer);

  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jkernelShape, JNI_ABORT);

  jlong returnValue = (jlong) network_layer;
  return returnValue;
}
