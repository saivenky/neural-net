#include <jni.h>
#include "neuron.h"
#include "saivenky_neural_ArrayCImplConvolutionLayer.h"

struct JIntArray {
  jintArray jarray;
  jint *array;
  jboolean isCopy;
};

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, void *address, int len) {
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

/*
 * Class:     saivenky_neural_ArrayCImplConvolutionLayer
 * Method:    createNativeLayer
 * Signature: ([I[III)J
 */
JNIEXPORT jlong JNICALL Java_saivenky_neural_ArrayCImplConvolutionLayer_createNativeLayer
  (JNIEnv * env, jobject object, jintArray inputShape, jintArray kernelShape, jint frames, jint stride) {

  struct JIntArray jinputShape, jkernelShape;
  jinputShape.jarray = inputShape;
  jkernelShape.jarray = kernelShape;
  GetIntArray(env, &jinputShape);
  GetIntArray(env, &jkernelShape);
  struct layer *layer = create_layer(jinputShape.array, jkernelShape.array, frames, stride);
  SetByteBuffer(env, object, "inputActivation", layer->inputActivation, layer->inputDim[LAST_DIM] * sizeof(double));
  SetByteBuffer(env, object, "inputError", layer->inputError, layer->inputDim[LAST_DIM] * sizeof(double));
  SetByteBuffer(env, object, "outputSignal", layer->outputSignal, layer->frames * layer->outputDim[LAST_DIM] * sizeof(double));
  SetByteBuffer(env, object, "outputError", layer->outputError, layer->frames * layer->outputDim[LAST_DIM] * sizeof(double));
  ReleaseIntArray(env, &jinputShape, JNI_ABORT);
  ReleaseIntArray(env, &jkernelShape, JNI_ABORT);
  jlong returnValue = (jlong) layer;
  return returnValue;
}

/*
 * Class:     saivenky_neural_ArrayCImplConvolutionLayer
 * Method:    applyConvolution
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_ArrayCImplConvolutionLayer_applyConvolution
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct layer *layer = (struct layer *)nativeLayerPtr;
    apply_kernel(layer);
  }

/*
 * Class:     saivenky_neural_ArrayCImplConvolutionLayer
 * Method:    backpropogateToProperties
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_ArrayCImplConvolutionLayer_backpropogateToProperties
  (JNIEnv *env, jobject object, jlong nativeLayerPtr) {
    struct layer *layer = (struct layer *)nativeLayerPtr;
    backpropogate_to_props(layer);
  }

/*
 * Class:     saivenky_neural_ArrayCImplConvolutionLayer
 * Method:    updateProperties
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_ArrayCImplConvolutionLayer_updateProperties
  (JNIEnv *env, jobject object, jlong nativeLayerPtr, jdouble rate) {
    struct layer *layer = (struct layer *)nativeLayerPtr;
    for(int i = 0; i < layer->frames; i++) {
      update_properties(layer->props[i], rate);
    }
  }
