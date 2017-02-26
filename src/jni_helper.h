#ifndef JNI_HELPER_H
#define JNI_HELPER_H

struct JIntArray {
  jintArray jarray;
  jint *array;
  jboolean isCopy;
};

struct JDoubleArray {
  jdoubleArray jarray;
  jdouble *array;
  jboolean isCopy;
};

void SetByteBuffer(JNIEnv *env, jobject obj, const char *fieldName, void *address, long len);
void GetIntArray(JNIEnv *env, struct JIntArray *array);
void ReleaseIntArray(JNIEnv *env, struct JIntArray *array, jint mode);
void GetDoubleArray(JNIEnv *env, struct JDoubleArray *array);
void ReleaseDoubleArray(JNIEnv *env, struct JDoubleArray *array, jint mode);
#endif
