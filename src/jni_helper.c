#include <jni.h>
#include <stdio.h>
#include "jni_helper.h"

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
