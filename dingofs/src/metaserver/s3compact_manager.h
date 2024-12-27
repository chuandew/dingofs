/*
 *  Copyright (c) 2021 NetEase Inc.
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
 * @Project: dingo
 * @Date: 2021-09-09
 * @Author: majie1
 */

#ifndef DINGOFS_SRC_METASERVER_S3COMPACT_MANAGER_H_
#define DINGOFS_SRC_METASERVER_S3COMPACT_MANAGER_H_

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "dingofs/proto/common.pb.h"
#include "dingofs/src/metaserver/s3compact.h"
#include "dingofs/src/metaserver/s3compact_worker.h"
#include "dingofs/src/metaserver/s3infocache.h"
#include "dingofs/src/utils/configuration.h"
#include "dingofs/src/utils/interruptible_sleeper.h"
#include "dingofs/src/aws/s3_adapter.h"

namespace dingofs {
namespace metaserver {

using dingofs::utils::Configuration;
using dingofs::utils::InterruptibleSleeper;
using dingofs::utils::RWLock;
using dingofs::aws::S3Adapter;
using dingofs::aws::S3AdapterOption;
using dingofs::common::S3Info;

class S3AdapterManager {
 private:
  std::mutex mtx_;
  bool inited_;
  uint64_t size_;  // same size as worker thread count
  std::vector<std::unique_ptr<S3Adapter>> s3adapters_;
  std::vector<bool> used_;
  S3AdapterOption opts_;

 public:
  explicit S3AdapterManager(uint64_t size, const S3AdapterOption& opts)
      : inited_(false), size_(size), opts_(opts) {}
  virtual ~S3AdapterManager() = default;
  virtual void Init();
  virtual void Deinit();
  virtual std::pair<uint64_t, S3Adapter*> GetS3Adapter();
  virtual void ReleaseS3Adapter(uint64_t index);
  virtual S3AdapterOption GetBasicS3AdapterOption();
};

struct S3CompactWorkQueueOption {
  S3AdapterOption s3opts;
  bool enable;
  uint64_t threadNum;
  uint64_t fragmentThreshold;
  uint64_t maxChunksPerCompact;
  uint64_t enqueueSleepMS;
  std::vector<std::string> mdsAddrs;
  std::string metaserverIpStr;
  uint64_t metaserverPort;
  uint64_t s3infocacheSize;
  uint64_t s3ReadMaxRetry;
  uint64_t s3ReadRetryInterval;

  void Init(std::shared_ptr<Configuration> conf);
};

class S3CompactManager {
 private:
  S3CompactWorkQueueOption opts_;
  std::unique_ptr<S3InfoCache> s3infoCache_;
  std::unique_ptr<S3AdapterManager> s3adapterManager_;

  S3CompactWorkerContext workerContext_;
  S3CompactWorkerOptions workerOptions_;

  std::vector<std::unique_ptr<S3CompactWorker>> workers_;

  bool inited_{false};

  S3CompactManager() = default;
  ~S3CompactManager() = default;

 public:
  static S3CompactManager& GetInstance() {
    static S3CompactManager instance_;
    return instance_;
  }

  void Init(std::shared_ptr<Configuration> conf);
  void Register(S3Compact s3compact);
  void Cancel(uint32_t partitionId);

  int Run();
  void Stop();
};

}  // namespace metaserver
}  // namespace dingofs

#endif  // DINGOFS_SRC_METASERVER_S3COMPACT_MANAGER_H_