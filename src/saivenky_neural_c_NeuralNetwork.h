/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class saivenky_neural_c_NeuralNetwork */

#ifndef _Included_saivenky_neural_c_NeuralNetwork
#define _Included_saivenky_neural_c_NeuralNetwork
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     saivenky_neural_c_NeuralNetwork
 * Method:    create
 * Signature: ([J)J
 */
JNIEXPORT jlong JNICALL Java_saivenky_neural_c_NeuralNetwork_create
  (JNIEnv *, jclass, jlongArray);

/*
 * Class:     saivenky_neural_c_NeuralNetwork
 * Method:    run
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_run
  (JNIEnv *, jclass, jlong);

/*
 * Class:     saivenky_neural_c_NeuralNetwork
 * Method:    update
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_update
  (JNIEnv *, jclass, jlong, jdouble);

/*
 * Class:     saivenky_neural_c_NeuralNetwork
 * Method:    train
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_saivenky_neural_c_NeuralNetwork_train
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif
