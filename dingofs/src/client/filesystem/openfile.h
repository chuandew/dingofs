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
 * Created Date: 2023-03-09
 * Author: Jingli Chen (Wine93)
 */

#ifndef DINGOFS_SRC_CLIENT_FILESYSTEM_OPENFILE_H_
#define DINGOFS_SRC_CLIENT_FILESYSTEM_OPENFILE_H_

#include <memory>
#include <unordered_map>

#include "dingofs/src/client/filesystem/defer_sync.h"
#include "dingofs/src/client/filesystem/dir_cache.h"
#include "dingofs/src/client/filesystem/meta.h"
#include "dingofs/src/client/filesystem/metric.h"
#include "dingofs/src/client/inode_wrapper.h"

namespace dingofs {
namespace client {
namespace filesystem {

using ::dingofs::utils::ReadLockGuard;
using ::dingofs::utils::WriteLockGuard;
using ::dingofs::client::InodeWrapper;
using ::dingofs::client::common::OpenFilesOption;

struct OpenFile {
  explicit OpenFile(std::shared_ptr<InodeWrapper> inode)
      : inode(inode), refs(0) {}

  std::shared_ptr<InodeWrapper> inode;
  uint64_t refs;
};

class OpenFiles {
 public:
  // option not used, but keeped here, we can turn on/off openfiles cache in the
  // future like juicefs
  explicit OpenFiles(OpenFilesOption option,
                     std::shared_ptr<DeferSync> defer_sync);

  void Open(Ino ino, std::shared_ptr<InodeWrapper> inode);

  bool IsOpened(Ino ino, std::shared_ptr<InodeWrapper>* inode);

  void Close(Ino ino);

  void CloseAll();

  bool GetFileAttr(Ino ino, InodeAttr* attr);

 private:
  RWLock rwlock_;
  OpenFilesOption option_;
  std::shared_ptr<DeferSync> deferSync_;
  std::unordered_map<Ino, std::unique_ptr<OpenFile>> files_;
  std::shared_ptr<OpenfilesMetric> metric_;
};

}  // namespace filesystem
}  // namespace client
}  // namespace dingofs

#endif  // DINGOFS_SRC_CLIENT_FILESYSTEM_OPENFILE_H_