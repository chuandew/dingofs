/*
 *  Copyright (c) 2023 NetEase Inc.
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
 * Project: Curve
 * Created Date: 2023-03-31
 * Author: Jingli Chen (Wine93)
 */

#ifndef DINGOFS_SRC_CLIENT_FILESYSTEM_LOOKUP_CACHE_H_
#define DINGOFS_SRC_CLIENT_FILESYSTEM_LOOKUP_CACHE_H_

#include <memory>
#include <string>

#include "dingofs/src/client/common/config.h"
#include "dingofs/src/client/filesystem/meta.h"
#include "dingofs/src/utils/lru_cache.h"

namespace dingofs {
namespace client {
namespace filesystem {

using ::dingofs::utils::LRUCache;
using ::dingofs::utils::ReadLockGuard;
using ::dingofs::utils::RWLock;
using ::dingofs::utils::WriteLockGuard;
using ::dingofs::client::common::LookupCacheOption;

// memory cache for lookup result, now we only support cache negative result,
// and other positive entry will be cached in kernel.
class LookupCache {
 public:
  struct CacheEntry {
    uint32_t uses;
    TimeSpec expireTime;
  };

  using LRUType = LRUCache<std::string, CacheEntry>;

 public:
  explicit LookupCache(LookupCacheOption option);

  bool Get(Ino parent, const std::string& name);

  bool Put(Ino parent, const std::string& name);

  bool Delete(Ino parent, const std::string& name);

 private:
  std::string CacheKey(Ino parent, const std::string& name);

 private:
  bool enable_;
  RWLock rwlock_;
  LookupCacheOption option_;
  std::shared_ptr<LRUType> lru_;
};

}  // namespace filesystem
}  // namespace client
}  // namespace dingofs

#endif  // DINGOFS_SRC_CLIENT_FILESYSTEM_LOOKUP_CACHE_H_