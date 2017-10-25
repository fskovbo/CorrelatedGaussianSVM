#ifndef MULTITHREADDRIVER_H
#define MULTITHREADDRIVER_H

#endif //REFACTOR_QPHYSICS_C_OPTIMIZATIONDRIVER_H
#include <thread>
#include <queue>
#include <mutex>
#include <iostream>

struct JobContainer{
public:
    size_t id         = 0;
    double oscWidth   = 1;
    size_t maxeval    = 1e5;
};

class MultithreadDriver{
private:
    int numThreads;
    std::queue<JobContainer> jobs;
public:
    MultithreadDriver(int numThreads);
    void addJob(JobContainer job);
    JobContainer getNextJob();
    void work(void (*f)(JobContainer job, size_t threadN));
};
