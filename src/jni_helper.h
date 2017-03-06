#ifndef JNI_HELPER_H
#define JNI_HELPER_H
#include <jni.h>
#include "network_layer.h"

struct JArray {
  jarray jarray;
  void *array;
  jboolean isCopy;
  JNIEnv *env;
  void * (*get)(JNIEnv *, jarray, jboolean *);
  void (*release)(JNIEnv *, jarray, void *, jint);
};

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, int index, void *address, long len);
struct JArray GetArray(JNIEnv *env, jarray,
  void * (*get)(JNIEnv *, jarray, jboolean *),
  void (*release)(JNIEnv *, jarray, void *, jint));
struct JArray GetIntArray(JNIEnv *env, jarray);
struct JArray GetDoubleArray(JNIEnv *env, jarray);
struct JArray GetLongArray(JNIEnv *env, jarray);
void ReleaseArray(struct JArray *array, jint mode);
void copy_network_layer_buffers(JNIEnv *env, jobject obj, struct network_layer *network_layer, int size);
#endif
