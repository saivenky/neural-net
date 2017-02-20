#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include "neuron_props.h"
#include "convolution_layer.h"
#include "sigmoid_layer.h"
#include "saivenky_neural_ConvolutionCImplLayer.h"

struct JIntArray {
  jintArray jarray;
  jint *array;
  jboolean isCopy;
};

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, void *address, long len) {
  jclass clazz = (*env)->GetObjectClass(env, obj);
  jfieldID fieldId = (*env)->GetFieldID(env, clazz, fieldName, "Ljava/nio/ByteBuffer;");
  jobject byteBuffer = (*env)->NewDirectByteBuffer(env, address, len);
  (*env)->SetObjectField(env, obj, fieldId, byteBuffer);
  printf("direct ByteByffer for '%s'\n", fieldName);
  fflush(stdout);
}
void GetIntArray(JNIEnv *env, struct JIntArray *array) {
  array->array = (*env)->GetIntArrayElements(env, array->jarray, &(array->isCopy));
}

void ReleaseIntArray(JNIEnv *env, struct JIntArray *array, jint mode) {
  if (array->isCopy == JNI_TRUE) {
    (*env)->ReleaseIntArrayElements(env, array->jarray, array->array, mode);
  }
}

struct bundle_layer {
  struct convolution_layer *conv;
  struct sigmoid_layer *sigm;
};

JNIEXPORT jlong JNICALL Java_saivenky_neural_ConvolutionCImplLayer_create
  (JNIEnv * env, jobject object, jintArray inputShape, jintArray kernelShape, jint frames, jint stride,
   jobject jinputActivation, jobject jinputError) {

  struct JIntArray jinputShape, jkernelShape;
  jinputShape.jarray = inputShape;
  jkernelShape.jarray = kernelShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jkernelShape);
  double *inputActivation = (jinputActivation == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputActivation);
  double *inputError = (jinputError == NULL) ? NULL : (*env)->GetDirectBufferAddress(env, jinputError);
  struct convolution_layer *layer1 = create_convolution_layer(jinputShape.array, jkernelShape.array, frames, stride, inputActivation, inputError);
  struct sigmoid_layer *layer2 = create_sigmoid_layer(layer1->frames * layer1->outputDim.dim2, layer1->outputSignal, layer1->outputError);
  SetByteBuffer(env, object, "outputSignal", layer2->outputSignal, layer2->size * sizeof(double));
  SetByteBuffer(env, object, "outputError", layer2->outputError, layer2->size * sizeof(double));
  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jkernelShape, JNI_ABORT);
  struct bundle_layer *layer = malloc(sizeof(struct bundle_layer));
  layer->conv = layer1;
  layer->sigm = layer2;
  jlong returnValue = (jlong) layer;
  return returnValue;
}

JNIEXPORT void JNICALL Java_saivenky_neural_ConvolutionCImplLayer_feedforward
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct bundle_layer *layer = (struct bundle_layer *)nativeLayerPtr;
    feedforward_convolution_layer(layer->conv);
    feedforward_sigmoid_layer(layer->sigm);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_ConvolutionCImplLayer_backpropogate
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct bundle_layer *layer = (struct bundle_layer *)nativeLayerPtr;
    backpropogate_sigmoid_layer(layer->sigm);
    backpropogate_convolution_layer(layer->conv);
  }

JNIEXPORT void JNICALL Java_saivenky_neural_ConvolutionCImplLayer_update
  (JNIEnv *env, jobject object, jlong nativeLayerPtr, jdouble rate) {
    struct bundle_layer *layer = (struct bundle_layer *)nativeLayerPtr;
    update_sigmoid_layer(layer->sigm);
    update_convolution_layer(layer->conv, rate);
  }
