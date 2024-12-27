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
 * Created Date: 2021-05-19
 * Author: chenwei
 */

#include "dingofs/src/mds/mds_service.h"

#include <vector>

namespace dingofs {
namespace mds {

using mds::Mountpoint;

void MdsServiceImpl::CreateFs(::google::protobuf::RpcController* controller,
                              const ::dingofs::mds::CreateFsRequest* request,
                              ::dingofs::mds::CreateFsResponse* response,
                              ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard doneGuard(done);
  const std::string& fsName = request->fsname();
  uint64_t blockSize = request->blocksize();
  FSType type = request->fstype();
  bool enableSumInDir = request->enablesumindir();

  // set response statuscode default value is ok
  response->set_statuscode(FSStatusCode::OK);

  LOG(INFO) << "CreateFs request: " << request->ShortDebugString();

  // create s3 fs
  auto createS3Fs = [&]() {
    if (!request->fsdetail().has_s3info()) {
      response->set_statuscode(FSStatusCode::PARAM_ERROR);
      LOG(ERROR) << "CreateFs request, type is s3, but has no s3info"
                 << ", fsName = " << fsName;
      return;
    }
    const auto& s3Info = request->fsdetail().s3info();
    FSStatusCode status =
        fsManager_->CreateFs(request, response->mutable_fsinfo());

    if (status != FSStatusCode::OK) {
      response->clear_fsinfo();
      response->set_statuscode(status);
      LOG(ERROR) << "CreateFs fail, fsName = " << fsName
                 << ", blockSize = " << blockSize
                 << ", s3Info.bucketname = " << s3Info.bucketname()
                 << ", enableSumInDir = " << enableSumInDir
                 << ", owner = " << request->owner()
                 << ", capacity = " << request->capacity()
                 << ", errCode = " << FSStatusCode_Name(status);
      return;
    }
  };

  switch (type) {
    case ::dingofs::common::FSType::TYPE_VOLUME:
      CHECK(false) << "CreateFs TYPE_VOLUME is not supported";
      break;
    case ::dingofs::common::FSType::TYPE_S3:
      createS3Fs();
      break;
    case ::dingofs::common::FSType::TYPE_HYBRID:
      CHECK(false) << "CreateFs TYPE_HYBRID is not supported";
      break;
    default:
      response->set_statuscode(FSStatusCode::PARAM_ERROR);
      LOG(ERROR) << "CreateFs fail, fs type is invalid"
                 << ", fsName = " << fsName << ", blockSize = " << blockSize
                 << ", fsType = " << type << ", errCode = "
                 << FSStatusCode_Name(FSStatusCode::PARAM_ERROR);
      break;
  }

  if (response->statuscode() != FSStatusCode::OK) {
    return;
  }
  LOG(INFO) << "CreateFs success, fsName = " << fsName
            << ", blockSize = " << blockSize << ", owner = " << request->owner()
            << ", capacity = " << request->capacity();
}

void MdsServiceImpl::MountFs(::google::protobuf::RpcController* controller,
                             const ::dingofs::mds::MountFsRequest* request,
                             ::dingofs::mds::MountFsResponse* response,
                             ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard doneGuard(done);
  const std::string& fsName = request->fsname();
  const Mountpoint& mount = request->mountpoint();
  LOG(INFO) << "MountFs request, fsName = " << fsName
            << ", mountPoint = " << mount.ShortDebugString();
  FSStatusCode status =
      fsManager_->MountFs(fsName, mount, response->mutable_fsinfo());
  if (status != FSStatusCode::OK) {
    response->clear_fsinfo();
    response->set_statuscode(status);
    LOG(ERROR) << "MountFs fail, fsName = " << fsName
               << ", mountPoint = " << mount.ShortDebugString()
               << ", errCode = " << FSStatusCode_Name(status);
    return;
  }

  response->set_statuscode(FSStatusCode::OK);
  LOG(INFO) << "MountFs success, fsName = " << fsName
            << ", mountPoint = " << mount.ShortDebugString()
            << ", mps: " << response->mutable_fsinfo()->mountpoints_size();
}

void MdsServiceImpl::UmountFs(::google::protobuf::RpcController* controller,
                              const ::dingofs::mds::UmountFsRequest* request,
                              ::dingofs::mds::UmountFsResponse* response,
                              ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard doneGuard(done);
  const std::string& fsName = request->fsname();
  const Mountpoint& mount = request->mountpoint();
  LOG(INFO) << "UmountFs request, " << request->ShortDebugString();
  FSStatusCode status = fsManager_->UmountFs(fsName, mount);
  if (status != FSStatusCode::OK) {
    response->set_statuscode(status);
    LOG(ERROR) << "UmountFs fail, fsName = " << fsName
               << ", mountPoint = " << mount.ShortDebugString()
               << ", errCode = " << FSStatusCode_Name(status);
    return;
  }

  response->set_statuscode(FSStatusCode::OK);
  LOG(INFO) << "UmountFs success, fsName = " << fsName
            << ", mountPoint = " << mount.ShortDebugString();
}

void MdsServiceImpl::GetFsInfo(::google::protobuf::RpcController* controller,
                               const ::dingofs::mds::GetFsInfoRequest* request,
                               ::dingofs::mds::GetFsInfoResponse* response,
                               ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard doneGuard(done);

  LOG(INFO) << "GetFsInfo request: " << request->ShortDebugString();

  FsInfo* fsInfo = response->mutable_fsinfo();
  FSStatusCode status = FSStatusCode::OK;
  if (request->has_fsid() && request->has_fsname()) {
    status = fsManager_->GetFsInfo(request->fsname(), request->fsid(), fsInfo);
  } else if (!request->has_fsid() && request->has_fsname()) {
    status = fsManager_->GetFsInfo(request->fsname(), fsInfo);
  } else if (request->has_fsid() && !request->has_fsname()) {
    status = fsManager_->GetFsInfo(request->fsid(), fsInfo);
  } else {
    status = FSStatusCode::PARAM_ERROR;
  }

  if (status != FSStatusCode::OK) {
    response->clear_fsinfo();
    response->set_statuscode(status);
    LOG(ERROR) << "GetFsInfo fail, request: " << request->ShortDebugString()
               << ", errCode = " << FSStatusCode_Name(status);
    return;
  }

  response->set_statuscode(FSStatusCode::OK);
  LOG(INFO) << "GetFsInfo success, response: " << response->ShortDebugString();
}

void MdsServiceImpl::DeleteFs(::google::protobuf::RpcController* controller,
                              const ::dingofs::mds::DeleteFsRequest* request,
                              ::dingofs::mds::DeleteFsResponse* response,
                              ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard doneGuard(done);
  const std::string& fsName = request->fsname();
  LOG(INFO) << "DeleteFs request, fsName = " << fsName;
  FSStatusCode status = fsManager_->DeleteFs(fsName);
  response->set_statuscode(status);
  if (status != FSStatusCode::OK && status != FSStatusCode::UNDER_DELETING) {
    LOG(ERROR) << "DeleteFs fail, fsName = " << fsName
               << ", errCode = " << FSStatusCode_Name(status);
    return;
  }

  LOG(INFO) << "DeleteFs success, fsName = " << fsName;
}

void MdsServiceImpl::AllocateS3Chunk(
    ::google::protobuf::RpcController* controller,
    const ::dingofs::mds::AllocateS3ChunkRequest* request,
    ::dingofs::mds::AllocateS3ChunkResponse* response,
    ::google::protobuf::Closure* done) {
  (void)controller;

  brpc::ClosureGuard guard(done);
  VLOG(9) << "start to allocate chunkId.";

  uint64_t chunkId = 0;
  uint64_t chunkIdNum = 1;
  if (request->has_chunkidnum()) {
    chunkIdNum = request->chunkidnum();
  }
  int stat = chunkIdAllocator_->GenChunkId(chunkIdNum, &chunkId);
  FSStatusCode resStat;
  if (stat >= 0) {
    resStat = OK;
  } else {
    resStat = ALLOCATE_CHUNKID_ERROR;
  }

  response->set_statuscode(resStat);

  if (resStat != OK) {
    LOG(ERROR) << "AllocateS3Chunk failure, request: "
               << request->ShortDebugString()
               << ", error: " << FSStatusCode_Name(resStat);
  } else {
    response->set_beginchunkid(chunkId);
    VLOG(9) << "AllocateS3Chunk success, request: "
            << request->ShortDebugString()
            << ", response: " << response->ShortDebugString();
  }
}

void MdsServiceImpl::ListClusterFsInfo(
    ::google::protobuf::RpcController* controller,
    const ::dingofs::mds::ListClusterFsInfoRequest* request,
    ::dingofs::mds::ListClusterFsInfoResponse* response,
    ::google::protobuf::Closure* done) {
  (void)controller;
  (void)request;

  brpc::ClosureGuard guard(done);
  LOG(INFO) << "start to check cluster fs info.";
  fsManager_->GetAllFsInfo(response->mutable_fsinfo());
  LOG(INFO) << "ListClusterFsInfo success, response: "
            << response->ShortDebugString();
}

void MdsServiceImpl::RefreshSession(
    ::google::protobuf::RpcController* controller,
    const ::dingofs::mds::RefreshSessionRequest* request,
    ::dingofs::mds::RefreshSessionResponse* response,
    ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard guard(done);
  fsManager_->RefreshSession(request, response);
  response->set_statuscode(FSStatusCode::OK);
}

void MdsServiceImpl::GetLatestTxId(
    ::google::protobuf::RpcController* controller,
    const GetLatestTxIdRequest* request, GetLatestTxIdResponse* response,
    ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard guard(done);
  VLOG(3) << "GetLatestTxId [request]: " << request->DebugString();
  fsManager_->GetLatestTxId(request, response);
  VLOG(3) << "GetLatestTxId [response]: " << response->DebugString();
}

void MdsServiceImpl::CommitTx(::google::protobuf::RpcController* controller,
                              const CommitTxRequest* request,
                              CommitTxResponse* response,
                              ::google::protobuf::Closure* done) {
  (void)controller;
  brpc::ClosureGuard guard(done);
  VLOG(3) << "CommitTx [request]: " << request->DebugString();
  fsManager_->CommitTx(request, response);
  VLOG(3) << "CommitTx [response]: " << response->DebugString();
}

}  // namespace mds
}  // namespace dingofs