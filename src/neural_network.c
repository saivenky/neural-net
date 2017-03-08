#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "neural_network.h"
#include "network_layer.h"

struct neural_network *create_neural_network(struct network_layer **layers, int size) {
  struct neural_network *nn = malloc(sizeof(struct neural_network));
  nn->size = size;
  nn->layers = malloc(nn->size * sizeof(struct network_layer *));
  nn->layers = memcpy(nn->layers, layers, nn->size * sizeof(struct network_layer *));

  int miniBatchSize = nn->layers[0]->miniBatchSize;
  nn->threads = malloc(miniBatchSize * sizeof(pthread_t));
  nn->thread_args = malloc(miniBatchSize * sizeof(struct neural_network_thread_args));

  for (int t = 0; t < miniBatchSize; t++) {
    nn->thread_args[t].nn = nn;
    nn->thread_args[t].batchIndex = t;
  }

  return nn;
}

void *feedforward_neural_network_thread(void *args) {
  struct neural_network_thread_args *thread_args = (struct neural_network_thread_args *)args;
  struct neural_network *nn = thread_args->nn;

  for (int i = 0; i < nn->size; i++) {
    feedforward_network_layer((void *)(nn->layers[i]), thread_args->batchIndex);
  }

  return NULL;
}

void thread_neural_network(struct neural_network *nn, void *(*thread_action)(void *)) {
  int miniBatchSize = nn->layers[0]->miniBatchSize;

  for (int t = 0; t < miniBatchSize; t++) {
    pthread_create(nn->threads + t, NULL, thread_action, nn->thread_args + t);
  }

  for (int t = 0; t < miniBatchSize; t++) {
    pthread_join(nn->threads[t], NULL);
  }
}

void feedforward_neural_network(struct neural_network *nn) {
  thread_neural_network(nn, &feedforward_neural_network_thread);
}

void *backpropogate_neural_network_thread(void *args) {
  struct neural_network_thread_args *thread_args = (struct neural_network_thread_args *)args;
  struct neural_network *nn = thread_args->nn;

  for (int i = nn->size - 1; i >= 0; i--) {
    backpropogate_network_layer((void *)(nn->layers[i]), thread_args->batchIndex);
  }

  return NULL;
}

void backpropogate_neural_network(struct neural_network *nn) {
  thread_neural_network(nn, &backpropogate_neural_network_thread);
}

void update_neural_network(struct neural_network *nn, float_t rate) {
  int miniBatchSize = nn->layers[0]->miniBatchSize;
  for (int t = 0; t < miniBatchSize; t++) {
    for (int i = 0; i < nn->size; i++) {
      update_network_layer((void *)(nn->layers[i]), rate, t);
    }
  }
}
