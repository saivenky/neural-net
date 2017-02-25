/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class saivenky_neural_c_ReluLayer */

#ifndef _Included_saivenky_neural_c_ReluLayer
#define _Included_saivenky_neural_c_ReluLayer
#ifdef __cplusplus
extern "C" {
#endif
#undef saivenky_neural_c_ReluLayer_SIZEOF_DOUBLE
#define saivenky_neural_c_ReluLayer_SIZEOF_DOUBLE 8L
/*
 * Class:     saivenky_neural_c_ReluLayer
 * Method:    create
 * Signature: (ILjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_create
  (JNIEnv *, jobject, jint, jobject, jobject);

/*
 * Class:     saivenky_neural_c_ReluLayer
 * Method:    destroy
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_saivenky_neural_c_ReluLayer_destroy
  (JNIEnv *, jobject, jlong);

/*
 * Class:     saivenky_neural_c_ReluLayer
 * Method:    feedforward
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_feedforward
  (JNIEnv *, jobject, jlong);

/*
 * Class:     saivenky_neural_c_ReluLayer
 * Method:    backpropogate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_backpropogate
  (JNIEnv *, jobject, jlong);

/*
 * Class:     saivenky_neural_c_ReluLayer
 * Method:    update
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_ReluLayer_update
  (JNIEnv *, jobject, jlong, jdouble);

#ifdef __cplusplus
}
#endif
#endif
