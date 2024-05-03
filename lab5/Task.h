#ifndef TASK_H
#define TASK_H
#include <cmath>

enum Tags {
    getTaskRequest = 0,
    getTaskResponse = 1,
};

struct Task {
    int weight;
};

double executeTask(Task &task) {
    double res = 0;
    for (int i = 0; i < task.weight; i++) {
        res += std::sqrt(i);
    }
    return res;
}
#endif