#include "MultithreadDriver.h"

MultithreadDriver::MultithreadDriver(int numThreads)
  : numThreads(numThreads){

}

void MultithreadDriver::addJob(JobContainer job){
  jobs.push(job);
}

JobContainer MultithreadDriver::getNextJob(){
  JobContainer thisJob = jobs.front();
  jobs.pop();
  return thisJob;
}

void MultithreadDriver::work(void (*f)(JobContainer job, size_t threadN)){
  while(!jobs.empty()){
      // Assign work for threads
      int threadID;
      std::vector<std::thread> threads;
      std::cout << "Main: \t\tInvoking threads\n";
      for (threadID = 0; threadID < numThreads; ++threadID) {
          if(jobs.empty()){
              break;
          } else{
              JobContainer currentJob = getNextJob();
              threads.push_back(std::thread(f,currentJob,threadID));
          }
      }
      // Wait for live threads to finish
      for (int i=0; i<threadID; ++i) {
          threads.at(i).join();
      }
  }
}
