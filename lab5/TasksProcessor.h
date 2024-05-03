#ifndef TASKS_PROCESSOR_H
#define TASKS_PROCESSOR_H
#include <pthread.h>
#include <cmath>
#include <deque>
#include "Task.h"
#include <string>
#include "Context.h"
using std::string;
using std::sqrt;
using std::deque;

void logProcessor(const string& message, int rank) {
    printf("rank: %d, Tasks Processor log: %s\n", rank, message.c_str());
}

void *processTasks(void *args) {
    Context context = *(Context *) args;
    deque<Task> &tasks = context.tasks;
    pthread_mutex_t &mutex = context.mutex;
    pthread_cond_t &needTasksCond = context.needTasksCond;
    int rank = context.rank;
    pthread_cond_t &availTasksCond = context.availTasksCond;
    int minimalQueueSize = context.minimalQueueSize;
    while (true) {

        pthread_mutex_lock(&mutex);

        if (tasks.size() < minimalQueueSize) {
            pthread_cond_signal(&needTasksCond);
        }

        if (tasks.empty()) {
            pthread_cond_wait(&availTasksCond, &mutex);
        }

        Task task = tasks.front();
        tasks.pop_front();

        pthread_mutex_unlock(&mutex);

        logProcessor("start of task", rank);
        executeTask(task);
    }
    return nullptr;
}

class TasksProcessor {
private:
    pthread_t thread{};

public:
    explicit TasksProcessor(pthread_attr_t &attr, Context &context) {
        pthread_create(&thread, &attr, processTasks, &context);
    }

    void join() const {
        pthread_join(thread, nullptr);
    }
};

#endif