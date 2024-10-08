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
 * Created Date: Thu Mar 21 2019
 * Author: xuchaojie
 */

#ifndef SRC_SNAPSHOTCLONESERVER_COMMON_TASK_INFO_H_
#define SRC_SNAPSHOTCLONESERVER_COMMON_TASK_INFO_H_

#include <atomic>
#include <memory>
#include <mutex>  //NOLINT
#include <string>

#include "src/common/concurrent/concurrent.h"

namespace curve {
namespace snapshotcloneserver {

class TaskInfo {
 public:
  TaskInfo() : progress_(0), isFinish_(false), isCanceled_(false) {}

  virtual ~TaskInfo() {}
  TaskInfo(const TaskInfo&) = delete;
  TaskInfo& operator=(const TaskInfo&) = delete;
  TaskInfo(TaskInfo&&) = default;
  TaskInfo& operator=(TaskInfo&&) = default;

  /**
   * @brief 设置任务完成度百分比
   *
   * @param persent 任务完成度百分比
   */
  void SetProgress(uint32_t persent) { progress_ = persent; }

  /**
   * @brief 获取任务完成度百分比
   *
   * @return 任务完成度百分比
   */
  uint32_t GetProgress() const { return progress_; }

  /**
   * @brief 完成任务
   */
  void Finish() { isFinish_.store(true); }

  /**
   * @brief 获取任务是否完成
   *
   * @retval true 任务完成
   * @retval false 任务未完成
   */
  bool IsFinish() const { return isFinish_.load(); }

  /**
   * @brief 取消任务
   */
  void Cancel() { isCanceled_ = true; }

  /**
   * @brief 获取任务是否取消
   *
   * @retval true 任务已取消
   * @retval false 任务未取消
   */
  bool IsCanceled() const { return isCanceled_; }

  /**
   * @brief 重置任务
   */
  void Reset() {
    isFinish_.store(false);
    isCanceled_ = false;
  }

  /**
   * @brief 获取任务锁的引用，以便使用LockGuard加锁解锁
   *
   *  用于同步任务完成和取消功能
   *  1. 任务完成前，先锁定任务，然后判断任务是否取消，
   *  若已取消，则释放锁，
   *  否则执行任务完成逻辑之后释放锁。
   *  2. 任务取消前，先锁定任务，然后判断任务是否完成，
   *  若已完成，则释放锁，
   *  否则执行任务取消逻辑之后释放锁。
   */
  curve::common::Mutex& GetLockRef() { return lock_; }

 private:
  // 任务完成度百分比
  uint32_t progress_;
  // 任务任务是否结束
  std::atomic_bool isFinish_;
  // 任务是否被取消
  bool isCanceled_;
  mutable curve::common::Mutex lock_;
};

}  // namespace snapshotcloneserver
}  // namespace curve

#endif  // SRC_SNAPSHOTCLONESERVER_COMMON_TASK_INFO_H_
