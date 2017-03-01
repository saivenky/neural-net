#include <jni.h>
#include <stdio.h>
#include "jni_helper.h"
#include "network_layer.h"

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, void *address, long len) {
  jclass clazz = (*env)->GetObjectClass(env, obj);
  jfieldID fieldId = (*env)->GetFieldID(env, clazz, fieldName, "Ljava/nio/ByteBuffer;");
  jobject byteBuffer = (*env)->NewDirectByteBuffer(env, address, len);
  (*env)->SetObjectField(env, obj, fieldId, byteBuffer);
}
void GetIntArray(JNIEnv *env, struct JIntArray *array) {
  array->array = (*env)->GetIntArrayElements(env, array->jarray, &(array->isCopy));
}

void ReleaseIntArray(JNIEnv *env, struct JIntArray *array, jint mode) {
  if (array->isCopy == JNI_TRUE) {
    (*env)->ReleaseIntArrayElements(env, array->jarray, array->array, mode);
  }
}

void GetDoubleArray(JNIEnv *env, struct JDoubleArray *array) {
  array->array = (*env)->GetDoubleArrayElements(env, array->jarray, &(array->isCopy));
}

void ReleaseDoubleArray(JNIEnv *env, struct JDoubleArray *array, jint mode) {
  if (array->isCopy == JNI_TRUE) {
    (*env)->ReleaseDoubleArrayElements(env, array->jarray, array->array, mode);
  }
}

void copy_network_layer_buffers(JNIEnv *env, jobject obj, struct network_layer *network_layer, int size) {
  SetByteBuffer(env, obj, "outputSignal", network_layer->activation.outputSignal, size * sizeof(double));
  SetByteBuffer(env, obj, "outputError", network_layer->gradient.outputError, size * sizeof(double));
}
