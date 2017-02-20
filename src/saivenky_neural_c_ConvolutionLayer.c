#include <jni.h>
#include "neuron_props.h"
#include "convolution_layer.h"
#include "saivenky_neural_c_ConvolutionLayer.h"
#include "jni_helper.h"

JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ConvolutionLayer_create
  (JNIEnv * env, jobject object, jintArray inputShape, jintArray kernelShape, jint frames, jint stride,
   jobject jinputActivation, jobject jinputError) {

  struct JIntArray jinputShape, jkernelShape;
  jinputShape.jarray = inputShape;
  jkernelShape.jarray = kernelShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jkernelShape);
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct convolution_layer *layer = create_convolution_layer(jinputShape.array, jkernelShape.array, frames, stride, inputActivation, inputError);
  SetByteBuffer(env, object, "outputSignal", layer->outputSignal, layer->frames * layer->outputDim.dim2 * sizeof(double));
  SetByteBuffer(env, object, "outputError", layer->outputError, layer->frames * layer->outputDim.dim2 * sizeof(double));
  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jkernelShape, JNI_ABORT);
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_feedforward
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct convolution_layer *layer = (struct convolution_layer *)nativeLayerPtr;
    feedforward_convolution_layer(layer);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_backpropogate
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct convolution_layer *layer = (struct convolution_layer *)nativeLayerPtr;
    backpropogate_convolution_layer(layer);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_c_ConvolutionLayer_update
  (JNIEnv *env, jobject object, jlong nativeLayerPtr, jdouble rate) {
    struct convolution_layer *layer = (struct convolution_layer *)nativeLayerPtr;
    update_convolution_layer(layer, rate);
  }
