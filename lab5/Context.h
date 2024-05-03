#ifndef CONTEXT_H
#define CONTEXT_H
#include <pthread.h>
#include "Task.h"
#include <deque>
using std::deque;

typedef struct Context {
    int size;
    int rank;
    int minimalQueueSize;
    deque<Task> &tasks;
    pthread_mutex_t &mutex;
    pthread_cond_t &availTasksCond;
    pthread_cond_t &needTasksCond;
} Context;

#endif //CONTEXT_H
