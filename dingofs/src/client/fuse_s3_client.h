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
 * Project: dingo
 * Created Date: Thur May 27 2021
 * Author: xuchaojie
 */

#ifndef DINGOFS_SRC_CLIENT_FUSE_S3_CLIENT_H_
#define DINGOFS_SRC_CLIENT_FUSE_S3_CLIENT_H_

#include <memory>

#include "brpc/server.h"
#include "dingofs/src/client/fuse_client.h"
#include "dingofs/src/client/s3/client_s3_cache_manager.h"
#include "dingofs/src/client/service/inode_objects_service.h"
#include "dingofs/src/client/warmup/warmup_manager.h"
#include "dingofs/src/aws/s3_adapter.h"

namespace dingofs {
namespace client {

using dingofs::aws::GetObjectAsyncCallBack;
using dingofs::aws::GetObjectAsyncContext;

namespace warmup {
class WarmupManager;
class WarmupManagerS3Impl;
}  // namespace warmup

class FuseS3Client : public FuseClient {
 public:
  FuseS3Client() : s3Adaptor_(std::make_shared<S3ClientAdaptorImpl>()) {
    auto read_func = [this](fuse_req_t req, fuse_ino_t ino, size_t size,
                            off_t off, struct fuse_file_info* fi, char* buffer,
                            size_t* r_size) {
      return FuseOpRead(req, ino, size, off, fi, buffer, r_size);
    };

    warmupManager_ = std::make_shared<warmup::WarmupManagerS3Impl>(
        metaClient_, inodeManager_, dentryManager_, fsInfo_, read_func,
        s3Adaptor_, nullptr);
  }

  FuseS3Client(const std::shared_ptr<MdsClient>& mdsClient,
               const std::shared_ptr<MetaServerClient>& metaClient,
               const std::shared_ptr<InodeCacheManager>& inodeManager,
               const std::shared_ptr<DentryCacheManager>& dentryManager,
               const std::shared_ptr<S3ClientAdaptor>& s3Adaptor,
               const std::shared_ptr<warmup::WarmupManager>& warmupManager)
      : FuseClient(mdsClient, metaClient, inodeManager, dentryManager,
                   warmupManager),
        s3Adaptor_(s3Adaptor) {}

  DINGOFS_ERROR Init(const FuseClientOption& option) override;

  void UnInit() override;

  DINGOFS_ERROR FuseOpInit(void* userdata,
                           struct fuse_conn_info* conn) override;

  DINGOFS_ERROR FuseOpWrite(fuse_req_t req, fuse_ino_t ino, const char* buf,
                            size_t size, off_t off, struct fuse_file_info* fi,
                            FileOut* file_out) override;

  DINGOFS_ERROR FuseOpRead(fuse_req_t req, fuse_ino_t ino, size_t size,
                           off_t off, struct fuse_file_info* fi, char* buffer,
                           size_t* r_size) override;

  DINGOFS_ERROR FuseOpCreate(fuse_req_t req, fuse_ino_t parent,
                             const char* name, mode_t mode,
                             struct fuse_file_info* fi,
                             EntryOut* entry_out) override;

  DINGOFS_ERROR FuseOpMkNod(fuse_req_t req, fuse_ino_t parent, const char* name,
                            mode_t mode, dev_t rdev,
                            EntryOut* entry_out) override;

  DINGOFS_ERROR FuseOpLink(fuse_req_t req, fuse_ino_t ino, fuse_ino_t newparent,
                           const char* newname, EntryOut* entry_out) override;

  DINGOFS_ERROR FuseOpUnlink(fuse_req_t req, fuse_ino_t parent,
                             const char* name) override;

  DINGOFS_ERROR FuseOpFsync(fuse_req_t req, fuse_ino_t ino, int datasync,
                            struct fuse_file_info* fi) override;

  DINGOFS_ERROR FuseOpFlush(fuse_req_t req, fuse_ino_t ino,
                            struct fuse_file_info* fi) override;

  DINGOFS_ERROR Truncate(InodeWrapper* inode, uint64_t length) override;

 private:
  bool InitKVCache(const KVClientManagerOpt& opt);

  void FlushData() override;

  DINGOFS_ERROR InitBrpcServer() override;

  // s3 adaptor
  std::shared_ptr<S3ClientAdaptor> s3Adaptor_;
  std::shared_ptr<KVClientManager> kvClientManager_;

  brpc::Server server_;
  InodeObjectsService inode_object_service_;
};

}  // namespace client
}  // namespace dingofs

#endif  // DINGOFS_SRC_CLIENT_FUSE_S3_CLIENT_H_