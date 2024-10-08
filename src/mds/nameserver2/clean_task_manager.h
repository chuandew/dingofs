/*
 *  Copyright (c) 2020 NetEase Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * Project: curve
 * Created Date: Tuesday December 18th 2018
 * Author: hzsunjianliang
 */
#ifndef SRC_MDS_NAMESERVER2_CLEAN_TASK_MANAGER_H_
#define SRC_MDS_NAMESERVER2_CLEAN_TASK_MANAGER_H_

#include <memory>
#include <mutex>   //NOLINT
#include <thread>  //NOLINT
#include <unordered_map>

#include "src/common/channel_pool.h"
#include "src/common/concurrent/concurrent.h"
#include "src/common/concurrent/task_thread_pool.h"
#include "src/common/interruptible_sleeper.h"
#include "src/mds/common/mds_define.h"
#include "src/mds/nameserver2/clean_task.h"

using ::curve::common::Atomic;
using ::curve::common::ChannelPool;
using ::curve::common::InterruptibleSleeper;

namespace curve {
namespace mds {

class CleanTaskManager {
 public:
  /**
   *  @brief 初始化TaskManager
   *  @param channelPool: 连接池
   *  @param threadNum: worker线程的数量
   *  @param checkPeriod: 周期性任务检查线程时间, ms
   */
  explicit CleanTaskManager(std::shared_ptr<ChannelPool> channelPool,
                            int threadNum = 10, int checkPeriod = 10000);
  ~CleanTaskManager() { Stop(); }

  /**
   * @brief 启动worker线程池、启动检查线程
   *
   */
  bool Start(void);

  /**
   * @brief 停止worker线程池、启动检查线程
   *
   */
  bool Stop(void);

  /**
   *  @brief 向线程池推送task
   *  @param task: 对应的工作任务
   *  @return 推送task是否成功，如已存在对应的任务，推送是吧
   */
  bool PushTask(std::shared_ptr<Task> task);

  /**
   * @brief 获取当前的task
   * @param id: 对应任务的相关文件InodeID
   * @return 返回对应task的shared_ptr 或者 不存在返回nullptr
   */
  std::shared_ptr<Task> GetTask(TaskIDType id);

 private:
  void CheckCleanResult(void);

 private:
  int threadNum_;
  ::curve::common::TaskThreadPool<>* cleanWorkers_;
  // for period check snapshot delete status
  std::unordered_map<TaskIDType, std::shared_ptr<Task>> cleanTasks_;
  common::Mutex mutex_;
  common::Thread* checkThread_;
  int checkPeriod_;

  Atomic<bool> stopFlag_;
  InterruptibleSleeper sleeper_;
  // 连接池，和chunkserverClient共享，没有任务在执行时清空
  std::shared_ptr<ChannelPool> channelPool_;
};

}  //  namespace mds
}  //  namespace curve

#endif  // SRC_MDS_NAMESERVER2_CLEAN_TASK_MANAGER_H_
