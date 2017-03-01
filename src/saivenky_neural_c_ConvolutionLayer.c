#include <jni.h>
#include "neuron_props.h"
#include "network_layer.h"
#include "convolution_layer.h"
#include "saivenky_neural_c_ConvolutionLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ConvolutionLayer_create
(JNIEnv * env, jobject obj, jintArray inputShape, jintArray kernelShape, jint frames, jint stride, jint padding,
 jobject jinputActivation, jobject jinputError) {

  struct JIntArray jinputShape, jkernelShape;
  jinputShape.jarray = inputShape;
  jkernelShape.jarray = kernelShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jkernelShape);
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);

  struct convolution_layer *layer = create_convolution_layer(jinputShape.array, jkernelShape.array, frames, stride, padding);
  struct network_layer *network_layer = create_network_layer(layer);
  network_layer->activation = create_activation_convolution_layer(layer, inputActivation);
  network_layer->gradient = create_gradient_convolution_layer(layer, inputError);
  copy_network_layer_buffers(env, obj, network_layer, layer->outputDim.dim2);

  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jkernelShape, JNI_ABORT);

  jlong returnValue = (jlong) network_layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_feedforward
(JNIEnv *env, jobject object, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  feedforward_convolution_layer(l->layer, l->activation);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_backpropogate
(JNIEnv *env, jobject object, jlong nativeLayerPtr) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  backpropogate_convolution_layer(l->layer, l->activation, l->gradient);
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_update
(JNIEnv *env, jobject object, jlong nativeLayerPtr, jdouble rate) {
  struct network_layer *l = (struct network_layer *)nativeLayerPtr;
  update_convolution_layer(l->layer, rate, l->gradient);
}
