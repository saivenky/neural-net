#include <jni.h>
#include "jni_helper.h"
#include "network_layer.h"

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, int index, void *address, long len) {
  jclass clazz = (*env)->GetObjectClass(env, obj);
  jfieldID fieldId = (*env)->GetFieldID(env, clazz, fieldName, "[Ljava/nio/ByteBuffer;");
  jobjectArray objArray = (*env)->GetObjectField(env, obj, fieldId);
  if (objArray == NULL) return;
  jobject byteBuffer = (*env)->NewDirectByteBuffer(env, address, len);
  (*env)->SetObjectArrayElement(env, objArray, index, byteBuffer);
}
struct JArray GetArray(JNIEnv *env, jarray array,
  void * (*get)(JNIEnv *, jarray, jboolean *),
  void (*release)(JNIEnv *, jarray, void *, jint)) {
  struct JArray result;
  result.env = env;
  result.jarray = array;
  result.get = get;
  result.release = release;
  result.array = get(env, result.jarray, &(result.isCopy));
  return result;
}

struct JArray GetIntArray(JNIEnv *env, jarray array) {
  return GetArray(env, array,
      (void *(*)(JNIEnv *,jarray,jboolean *))(*env)->GetIntArrayElements,
      (void (*)(JNIEnv *,jarray,void  *,jint))(*env)->ReleaseIntArrayElements);
}

struct JArray GetDoubleArray(JNIEnv *env, jarray array) {
  return GetArray(env, array,
      (void *(*)(JNIEnv *,jarray,jboolean *))(*env)->GetDoubleArrayElements,
      (void (*)(JNIEnv *,jarray,void  *,jint))(*env)->ReleaseDoubleArrayElements);
}

struct JArray GetLongArray(JNIEnv *env, jarray array) {
  return GetArray(env, array,
      (void *(*)(JNIEnv *,jarray,jboolean *))(*env)->GetLongArrayElements,
      (void (*)(JNIEnv *,jarray,void  *,jint))(*env)->ReleaseLongArrayElements);
}

void ReleaseArray(struct JArray *array, jint mode) {
  if (array->isCopy == JNI_TRUE) {
    array->release(array->env, array->jarray, array->array, mode);
  }
}

void copy_network_layer_buffers(JNIEnv *env, jobject obj, struct network_layer *network_layer, int size) {
  for (int i = 0; i < network_layer->miniBatchSize; i++) {
    SetByteBuffer(env, obj, "outputSignals", i, network_layer->activations[i].outputSignal, size * sizeof(double));
    SetByteBuffer(env, obj, "outputErrors", i, network_layer->gradients[i].outputError, size * sizeof(double));
  }
}
