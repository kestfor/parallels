#ifndef TASKS_LISTENER_H
#define TASKS_LISTENER_H

#include <pthread.h>
#include "Context.h"
#include <mpi.h>
#include "Task.h"
#include <string>
#include <deque>
using std::string;
using std::deque;

void *listen(void *args) {
    Context context = *(Context *) args;
    deque<Task> &tasks = context.tasks;
    pthread_mutex_t &mutex = context.mutex;
    pthread_cond_t &availableTaskCond = context.availTasksCond;
    while (true) {
        MPI_Request request;
        MPI_Status status;

        int data;
        MPI_Irecv(&data, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);

        if (status.MPI_TAG == Tags::getTaskRequest) {
            //send task to process
            pthread_mutex_lock(&mutex);
            Task taskBuff{};
            if (tasks.size() > context.minimalQueueSize - 1) {
                taskBuff = tasks.back();
                tasks.pop_back();
            } else {
                taskBuff = {-1};
            }

            pthread_mutex_unlock(&mutex);

            int tag = Tags::getTaskResponse;
            int to = status.MPI_SOURCE;


            MPI_Isend(&taskBuff, 1, MPI_INT, to, tag, MPI_COMM_WORLD, &request);
        } else if (status.MPI_TAG == Tags::getTaskResponse) {
            //receive task from process
            if (data != -1) {
                pthread_mutex_lock(&mutex);

                if (tasks.empty()) {
                    tasks.push_back({data});
                    pthread_cond_signal(&availableTaskCond);
                } else {
                    tasks.push_back({data});
                }
                pthread_mutex_unlock(&mutex);
            }
        }
    }
    return nullptr;
}

class MessageHandler {
private:
    pthread_t thread{};

public:
    explicit MessageHandler(pthread_attr_t &attr, Context &context) {
        pthread_create(&thread, &attr, listen, &context);
    }

    void join() const {
        pthread_join(thread, nullptr);
    }
};
#endif