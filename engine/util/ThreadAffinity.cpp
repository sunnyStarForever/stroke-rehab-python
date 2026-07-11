#include "engine/util/ThreadAffinity.h"

#include <string>

#include "engine/util/Logger.h"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace rehab {

bool bindCurrentThreadToCpu(int cpuId, const char* threadName) {
#ifdef __linux__
  const std::string name =
      (threadName != nullptr && *threadName != '\0') ? threadName : "thread";
  if (cpuId < 0 || cpuId >= CPU_SETSIZE) {
    Logger::warn("bind thread " + name + " to CPU" + std::to_string(cpuId) +
                 " failed");
    return false;
  }

  cpu_set_t cpuSet;
  CPU_ZERO(&cpuSet);
  CPU_SET(cpuId, &cpuSet);

  const int rc =
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
  if (rc == 0) {
    Logger::info("bind thread " + name + " to CPU" + std::to_string(cpuId));
    return true;
  }

  Logger::warn("bind thread " + name + " to CPU" + std::to_string(cpuId) +
               " failed");
  return false;
#else
  (void)cpuId;
  (void)threadName;
  return false;
#endif
}

}  // namespace rehab
