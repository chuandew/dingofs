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
 * File Created: 2022-06-30
 * Author: xuchaojie
 */

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/client/chunkserver_client.h"
#include "src/common/task_tracker.h"
#include "test/client/mock/mock_chunkservice.h"

using ::curve::chunkserver::CHUNK_OP_STATUS;
using ::curve::common::TaskTracker;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace curve {
namespace client {

class ChunkServerClientTest : public testing::Test {
 protected:
  virtual void SetUp() {
    listenAddr_ = "chunkserverclienttest_cs_listenAddr";
    server_ = new brpc::Server();
  }

  virtual void TearDown() {
    server_->Stop(0);
    server_->Join();
    delete server_;
    server_ = nullptr;
  }

 public:
  std::string listenAddr_;
  brpc::Server* server_;
};

struct FakeUpdateFileEpochClosure : public ChunkServerClientClosure {
 public:
  FakeUpdateFileEpochClosure() : runCnt_(0) {}
  ~FakeUpdateFileEpochClosure() {}
  void Run() override {
    runCnt_++;
    tracker_->HandleResponse(GetErrorCode());
  }

  int GetRunCnt() { return runCnt_; }

  void AddToBeTraced(const std::shared_ptr<TaskTracker>& tracker) {
    tracker->AddOneTrace();
    tracker_ = tracker;
  }

 private:
  int runCnt_;
  std::shared_ptr<TaskTracker> tracker_;
};

TEST_F(ChunkServerClientTest, UpdateFileEpochSuccess) {
  MockChunkServiceImpl mockChunkService;
  ASSERT_EQ(
      server_->AddService(&mockChunkService, brpc::SERVER_DOESNT_OWN_SERVICE),
      0);
  ASSERT_EQ(server_->Start(listenAddr_.c_str(), nullptr), 0);

  CHUNK_OP_STATUS csRet = CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS;
  EXPECT_CALL(mockChunkService, UpdateEpoch(_, _, _, _))
      .WillOnce(
          Invoke([=](::google::protobuf::RpcController* controller,
                     const ::curve::chunkserver::UpdateEpochRequest* request,
                     ::curve::chunkserver::UpdateEpochResponse* response,
                     google::protobuf::Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            response->set_status(csRet);
          }));

  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;
  EndPoint ep;
  str2endpoint(listenAddr_.c_str(), &ep);
  cs.internalAddr = PeerAddr(ep);
  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(0, ret);

  ASSERT_EQ(0, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

TEST_F(ChunkServerClientTest, UpdateFileEpochSuccessUsingExternalIp) {
  MockChunkServiceImpl mockChunkService;
  ASSERT_EQ(
      server_->AddService(&mockChunkService, brpc::SERVER_DOESNT_OWN_SERVICE),
      0);
  ASSERT_EQ(server_->Start(listenAddr_.c_str(), nullptr), 0);

  CHUNK_OP_STATUS csRet = CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS;
  EXPECT_CALL(mockChunkService, UpdateEpoch(_, _, _, _))
      .WillOnce(
          Invoke([=](::google::protobuf::RpcController* controller,
                     const ::curve::chunkserver::UpdateEpochRequest* request,
                     ::curve::chunkserver::UpdateEpochResponse* response,
                     google::protobuf::Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            response->set_status(csRet);
          }));

  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;

  EndPoint ep;
  str2endpoint("", &ep);
  cs.internalAddr = PeerAddr(ep);

  EndPoint ep2;
  str2endpoint(listenAddr_.c_str(), &ep2);
  cs.externalAddr = PeerAddr(ep2);

  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(0, ret);

  ASSERT_EQ(0, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

TEST_F(ChunkServerClientTest, UpdateFileEpochSuccessForChunkServerOffline) {
  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;
  EndPoint ep;
  str2endpoint(listenAddr_.c_str(), &ep);
  cs.internalAddr = PeerAddr(ep);
  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(0, ret);

  ASSERT_EQ(0, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

TEST_F(ChunkServerClientTest, UpdateFileEpochFailedByEpochTooOld) {
  MockChunkServiceImpl mockChunkService;
  ASSERT_EQ(
      server_->AddService(&mockChunkService, brpc::SERVER_DOESNT_OWN_SERVICE),
      0);
  ASSERT_EQ(server_->Start(listenAddr_.c_str(), nullptr), 0);

  CHUNK_OP_STATUS csRet = CHUNK_OP_STATUS::CHUNK_OP_STATUS_EPOCH_TOO_OLD;
  EXPECT_CALL(mockChunkService, UpdateEpoch(_, _, _, _))
      .WillOnce(
          Invoke([=](::google::protobuf::RpcController* controller,
                     const ::curve::chunkserver::UpdateEpochRequest* request,
                     ::curve::chunkserver::UpdateEpochResponse* response,
                     google::protobuf::Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            response->set_status(csRet);
          }));

  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;
  EndPoint ep;
  str2endpoint(listenAddr_.c_str(), &ep);
  cs.internalAddr = PeerAddr(ep);
  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(-LIBCURVE_ERROR::EPOCH_TOO_OLD, ret);

  ASSERT_EQ(-LIBCURVE_ERROR::EPOCH_TOO_OLD, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

TEST_F(ChunkServerClientTest, UpdateFileEpochFailedUnknown) {
  MockChunkServiceImpl mockChunkService;
  ASSERT_EQ(
      server_->AddService(&mockChunkService, brpc::SERVER_DOESNT_OWN_SERVICE),
      0);
  ASSERT_EQ(server_->Start(listenAddr_.c_str(), nullptr), 0);

  CHUNK_OP_STATUS csRet = CHUNK_OP_STATUS::CHUNK_OP_STATUS_FAILURE_UNKNOWN;
  EXPECT_CALL(mockChunkService, UpdateEpoch(_, _, _, _))
      .WillOnce(
          Invoke([=](::google::protobuf::RpcController* controller,
                     const ::curve::chunkserver::UpdateEpochRequest* request,
                     ::curve::chunkserver::UpdateEpochResponse* response,
                     google::protobuf::Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            response->set_status(csRet);
          }));

  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;
  EndPoint ep;
  str2endpoint(listenAddr_.c_str(), &ep);
  cs.internalAddr = PeerAddr(ep);
  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(-LIBCURVE_ERROR::UNKNOWN, ret);

  ASSERT_EQ(-LIBCURVE_ERROR::UNKNOWN, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

TEST_F(ChunkServerClientTest, UpdateFileEpochFailedForRetryTimesExceed) {
  MockChunkServiceImpl mockChunkService;
  ASSERT_EQ(
      server_->AddService(&mockChunkService, brpc::SERVER_DOESNT_OWN_SERVICE),
      0);
  ASSERT_EQ(server_->Start(listenAddr_.c_str(), nullptr), 0);

  CHUNK_OP_STATUS csRet = CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS;
  EXPECT_CALL(mockChunkService, UpdateEpoch(_, _, _, _))
      .WillRepeatedly(
          Invoke([=](::google::protobuf::RpcController* controller,
                     const ::curve::chunkserver::UpdateEpochRequest* request,
                     ::curve::chunkserver::UpdateEpochResponse* response,
                     google::protobuf::Closure* done) {
            // sleep a while to ensure client side timeout
            bthread_usleep(1000000);
            brpc::ClosureGuard doneGuard(done);
            response->set_status(csRet);
          }));

  CopysetPeerInfo<ChunkServerID> cs;
  cs.peerID = 1;
  EndPoint ep;
  str2endpoint(listenAddr_.c_str(), &ep);
  cs.internalAddr = PeerAddr(ep);
  uint64_t fileId = 1;
  uint64_t epoch = 1;
  auto tracker = std::make_shared<TaskTracker>();
  FakeUpdateFileEpochClosure* closure = new FakeUpdateFileEpochClosure();

  ChunkServerClient csClient;
  ChunkServerClientRetryOptions ops;
  ops.rpcMaxTry = 3;
  ops.rpcTimeoutMs = 1;
  csClient.Init(ops);
  int ret = csClient.UpdateFileEpoch(cs, fileId, epoch, closure);
  ASSERT_EQ(0, ret);

  closure->AddToBeTraced(tracker);
  tracker->Wait();
  ret = tracker->GetResult();
  ASSERT_EQ(-LIBCURVE_ERROR::FAILED, ret);

  ASSERT_EQ(-LIBCURVE_ERROR::FAILED, closure->GetErrorCode());
  ASSERT_EQ(1, closure->GetRunCnt());
}

}  // namespace client
}  // namespace curve

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::InitGoogleMock(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, false);

  int ret = RUN_ALL_TESTS();
  return ret;
}
