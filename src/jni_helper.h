#ifndef JNI_HELPER_H
#define JNI_HELPER_H

struct JIntArray {
  jintArray jarray;
  jint *array;
  jboolean isCopy;
};

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, void *address, long len);
void GetIntArray(JNIEnv *env, struct JIntArray *array);
void ReleaseIntArray(JNIEnv *env, struct JIntArray *array, jint mode);
#endif
