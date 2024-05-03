#ifndef TASKS_REQUESTER_H
#define TASKS_REQUESTER_H
#include <pthread.h>
#include <mpi.h>
#include "Task.h"
#include <deque>
#include <cstring>
#include <string>
#include "Context.h"
using std::string;
using std::deque;

static void getTasksRequest(int size, int rank) {
    MPI_Request reqsSend[size];
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            MPI_Isend(nullptr, 0, MPI_INT, i, Tags::getTaskRequest, MPI_COMM_WORLD, &reqsSend[i]);
        }
    }
}

static void *processRequester(void *args) {
    Context context = *(Context *) args;
    deque<Task> &tasks = context.tasks;
    pthread_mutex_t &mutex = context.mutex;
    pthread_cond_t &needTasksCond = context.needTasksCond;
    int minimalQueueSize = context.minimalQueueSize;
    int size = context.size;
    int rank = context.rank;

    pthread_mutex_lock(&mutex);
    bool needTasks = tasks.size() < minimalQueueSize;
    pthread_mutex_unlock(&mutex);
    while (true) {
        pthread_mutex_lock(&mutex);
        if (!needTasks) {
            pthread_cond_wait(&needTasksCond, &mutex);
        }
        getTasksRequest(size, rank);
        needTasks = false;
        pthread_mutex_unlock(&mutex);
    }
    return nullptr;
}

class TasksRequester {
private:
    pthread_t thread{};

public:
    explicit TasksRequester(pthread_attr_t &attr, Context &context) {
        pthread_create(&thread, &attr, processRequester, &context);
    }

    void join() const {
        pthread_join(thread, nullptr);
    }

};
#endif