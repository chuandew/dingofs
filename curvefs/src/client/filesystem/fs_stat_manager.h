// Copyright (c) 2024 dingodb.com, Inc. All Rights Reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CURVEFS_SRC_FILE_SYSTEM_STAT_MANAGER_H_
#define CURVEFS_SRC_FILE_SYSTEM_STAT_MANAGER_H_

#include <atomic>
#include <memory>

#include "curvefs/src/base/timer/timer.h"
#include "curvefs/src/client/filesystem/dir_quota.h"

namespace curvefs {
namespace client {
namespace filesystem {

using base::timer::Timer;
using rpcclient::MetaServerClient;

class FsStatManager {
 public:
  FsStatManager(uint32_t fs_id, std::shared_ptr<MetaServerClient> meta_client,
                std::shared_ptr<Timer> timer)
      : fs_id_(fs_id), meta_client_(meta_client), timer_(std::move(timer)) {}

  virtual ~FsStatManager() = default;

  void Start();

  void Stop();

  bool IsRunning() const { return running_.load(); }

  void UpdateQuotaUsage(int64_t new_space, int64_t new_inodes);

  bool CheckQuota(int64_t new_space, int64_t new_inodes);

 private:
  void InitQuota();
  CURVEFS_ERROR LoadFsQuota();

  void FlushFsUsage();
  void DoFlushFsUsage();

  const uint32_t fs_id_{0};
  std::shared_ptr<MetaServerClient> meta_client_;
  std::shared_ptr<Timer> timer_;

  std::atomic<bool> running_{false};

  std::unique_ptr<DirQuota> fs_quota_;
};

}  // namespace filesystem
}  // namespace client
}  // namespace curvefs

#endif  // CURVEFS_SRC_FILE_SYSTEM_STAT_MANAGER_H_