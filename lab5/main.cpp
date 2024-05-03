#include <pthread.h>
#include <cstdio>
#include "MessageHandler.h"
#include "TasksProcessor.h"
#include "TasksRequester.h"
#include <deque>
#include <random>
using std::deque;

void generateTasks(deque<Task> &tasks, int rank) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distributionForWeights(10e+4, 10e+9);
    std::uniform_int_distribution<std::mt19937::result_type> distributionForNums(10, 20);
    if (rank % 2 == 0) {
        for (int j = 0; j < distributionForNums(rng); j++) {
            tasks.push_back({(int) distributionForWeights(rng)});
        }
    }
}

int main(int argc, char *argv[]) {
    int provided_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level);
    if (provided_level != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        perror("Couldn't provide thread multiple level\n");
        exit(-1);
    }
    int size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    pthread_attr_t attr;
    pthread_mutex_t mutex;
    pthread_cond_t needCond;
    pthread_cond_t emptyCond;

    pthread_attr_init(&attr);
    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&needCond, nullptr);
    pthread_cond_init(&emptyCond, nullptr);

    deque<Task> tasks;
    generateTasks(tasks, rank);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Context ctx = {size, rank, 2, tasks, mutex, emptyCond, needCond};


    TasksProcessor taskProcessor(attr, ctx);
    MessageHandler messageHandler(attr, ctx);
    TasksRequester tasksRequester(attr, ctx);


    tasksRequester.join();
    messageHandler.join();
    taskProcessor.join();

    pthread_cond_destroy(&emptyCond);
    pthread_cond_destroy(&needCond);
    pthread_mutex_destroy(&mutex);
    pthread_attr_destroy(&attr);
    MPI_Finalize();
}