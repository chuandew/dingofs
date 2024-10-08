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
 * File Created: Monday, 17th September 2018 4:15:27 pm
 * Author: tongguangxun
 */

#ifndef SRC_CLIENT_IOMANAGER4FILE_H_
#define SRC_CLIENT_IOMANAGER4FILE_H_

#include <bthread/condition_variable.h>
#include <bthread/mutex.h>

#include <atomic>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <string>

#include "include/curve_compiler_specific.h"
#include "src/client/client_common.h"
#include "src/client/discard_task.h"
#include "src/client/inflight_controller.h"
#include "src/client/iomanager.h"
#include "src/client/mds_client.h"
#include "src/client/metacache.h"
#include "src/client/request_scheduler.h"
#include "src/common/concurrent/concurrent.h"
#include "src/common/concurrent/task_thread_pool.h"
#include "src/common/throttle.h"

namespace curve {
namespace client {

using curve::common::Atomic;

class FlightIOGuard;

class IOManager4File : public IOManager {
 public:
  IOManager4File();
  ~IOManager4File() = default;

  /**
   * 初始化函数
   * @param: filename为当前iomanager服务的文件名
   * @param: ioopt为当前iomanager的配置信息
   * @param: mdsclient向下透传给metacache
   * @return: 成功true,失败false
   */
  bool Initialize(const std::string& filename, const IOOption& ioOpt,
                  MDSClient* mdsclient);

  /**
   * @brief Recycle resources
   */
  void UnInitialize();

  /**
   * 同步模式读
   * @param: buf为当前待读取的缓冲区
   * @param：offset文件内的便宜
   * @parma：length为待读取的长度
   * @param: mdsclient透传给底层，在必要的时候与mds通信
   * @return: 成功返回读取真实长度，-1为失败
   */
  int Read(char* buf, off_t offset, size_t length, MDSClient* mdsclient);
  /**
   * 同步模式写
   * @param: mdsclient透传给底层，在必要的时候与mds通信
   * @param: buf为当前待写入的缓冲区
   * @param：offset文件内的便宜
   * @param：length为待读取的长度
   * @return： 成功返回写入真实长度，-1为失败
   */
  int Write(const char* buf, off_t offset, size_t length, MDSClient* mdsclient);
  /**
   * 异步模式读
   * @param: mdsclient透传给底层，在必要的时候与mds通信
   * @param: aioctx为异步读写的io上下文，保存基本的io信息
   * @param dataType type of aioctx->buf
   * @return： 0为成功，小于0为失败
   */
  int AioRead(CurveAioContext* aioctx, MDSClient* mdsclient,
              UserDataType dataType);
  /**
   * 异步模式写
   * @param: mdsclient透传给底层，在必要的时候与mds通信
   * @param: aioctx为异步读写的io上下文，保存基本的io信息
   * @param dataType type of aioctx->buf
   * @return： 0为成功，小于0为失败
   */
  int AioWrite(CurveAioContext* aioctx, MDSClient* mdsclient,
               UserDataType dataType);

  /**
   * @brief Synchronous discard operation
   * @param offset discard offset
   * @param length discard length
   * @return On success, returns 0.
   *         On error, returns a negative value.
   */
  int Discard(off_t offset, size_t length, MDSClient* mdsclient);

  /**
   * @brief Asynchronous discard operation
   * @param aioctx async request context
   * @param mdsclient for communicate with MDS
   * @return 0 means success, otherwise it means failure
   */
  int AioDiscard(CurveAioContext* aioctx, MDSClient* mdsclient);

  /**
   * @brief 获取rpc发送令牌
   */
  void GetInflightRpcToken() override;

  /**
   * @brief 释放rpc发送令牌
   */
  void ReleaseInflightRpcToken() override;

  /**
   * 获取metacache，测试代码使用
   */
  MetaCache* GetMetaCache() { return &mc_; }
  /**
   * 设置scheduler，测试代码使用
   */
  void SetRequestScheduler(RequestScheduler* scheduler) {
    scheduler_ = scheduler;
  }

  /**
   * 获取metric信息，测试代码使用
   */
  FileMetric* GetMetric() { return fileMetric_; }

  /**
   * 重新设置io配置信息，测试使用
   */
  void SetIOOpt(const IOOption& opt) { ioopt_ = opt; }

  /**
   * 测试使用，获取request scheduler
   */
  RequestScheduler* GetScheduler() { return scheduler_; }

  /**
   * lease excutor在检查到版本更新的时候，需要通知iomanager更新文件版本信息
   * @param: fi为当前需要更新的文件信息
   */
  void UpdateFileInfo(const FInfo_t& fi);

  const FInfo* GetFileInfo() const { return mc_.GetFileInfo(); }

  void UpdateFileEpoch(const FileEpoch& fEpoch) { mc_.UpdateFileEpoch(fEpoch); }

  const FileEpoch* GetFileEpoch() const { return mc_.GetFileEpoch(); }

  /**
   * 返回文件最新版本号
   */
  uint64_t GetLatestFileSn() const { return mc_.GetLatestFileSn(); }

  /**
   * 更新文件最新版本号
   */
  void SetLatestFileSn(uint64_t newSn) { mc_.SetLatestFileSn(newSn); }

  /**
   * @brief get current file inodeid
   * @return file inodeid
   */
  uint64_t InodeId() const { return mc_.InodeId(); }

  void UpdateFileThrottleParams(const common::ReadWriteThrottleParams& params);

  void SetDisableStripe();

 private:
  friend class LeaseExecutor;
  friend class FlightIOGuard;
  /**
   * lease相关接口，当LeaseExecutor续约失败的时候，调用LeaseTimeoutDisableIO
   * 将新下发的IO全部失败返回
   */
  void LeaseTimeoutBlockIO();

  /**
   * 当lease又续约成功的时候，LeaseExecutor调用该接口恢复IO
   */
  void ResumeIO();

  /**
   * 当lesaeexcutor发现版本变更，调用该接口开始等待inflight回来，这段期间IO是hang的
   */
  void BlockIO();

  /**
   * 因为curve client底层都是异步IO，每个IO会分配一个IOtracker跟踪IO
   * 当这个IO做完之后，底层需要告知当前io manager来释放这个IOTracker，
   * HandleAsyncIOResponse负责释放IOTracker
   * @param: iotracker是返回的异步io
   */
  void HandleAsyncIOResponse(IOTracker* iotracker) override;

  class FlightIOGuard {
   public:
    explicit FlightIOGuard(IOManager4File* iomana) {
      iomanager = iomana;
      iomanager->inflightCntl_.IncremInflightNum();
    }

    ~FlightIOGuard() { iomanager->inflightCntl_.DecremInflightNum(); }

   private:
    IOManager4File* iomanager;
  };

  bool IsNeedDiscard(size_t len) const;

 private:
  // 每个IOManager都有其IO配置，保存在iooption里
  IOOption ioopt_;

  // metacache存储当前文件的所有元数据信息
  MetaCache mc_;

  // IO最后由schedule模块向chunkserver端分发，scheduler由IOManager创建和释放
  RequestScheduler* scheduler_;

  // client端metric统计信息
  FileMetric* fileMetric_;

  // task thread pool为了将qemu线程与curve线程隔离
  curve::common::TaskThreadPool<bthread::Mutex, bthread::ConditionVariable>
      taskPool_;

  // inflight IO控制
  InflightControl inflightCntl_;

  // inflight rpc控制
  InflightControl inflightRpcCntl_;

  std::unique_ptr<common::Throttle> throttle_;

  // 是否退出
  bool exit_;

  // lease续约线程与qemu一侧线程调用是并发的
  // qemu在调用close的时候会关闭iomanager及其对应
  // 资源。lease续约线程在续约成功或失败的时候会通知iomanager的
  // scheduler线程现在需要block IO或者resume IO，所以
  // 如果在lease续约线程需要通知iomanager的时候，这时候
  // 如果iomanager的资源scheduler已经被释放了，就会
  // 导致crash，所以需要对这个资源加一把锁，在退出的时候
  // 不会有并发的情况，保证在资源被析构的时候lease续约
  // 线程不会再用到这些资源.
  std::mutex exitMtx_;

  // enable/disable stripe for read/write of stripe file
  // currently only one scenario set this field to true:
  // chunkserver use client to read clone source file,
  // because client's IO requests already transformed by stripe parameters
  bool disableStripe_;

  std::unique_ptr<DiscardTaskManager> discardTaskManager_;
};

}  // namespace client
}  // namespace curve
#endif  // SRC_CLIENT_IOMANAGER4FILE_H_
