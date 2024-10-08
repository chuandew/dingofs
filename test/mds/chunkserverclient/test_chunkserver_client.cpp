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
 * Created Date: Wed Mar 13 2019
 * Author: xuchaojie
 */

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>  //NOLINT
#include <thread>  //NOLINT

#include "proto/chunk.pb.h"
#include "proto/cli.pb.h"
#include "src/mds/chunkserverclient/chunkserver_client.h"
#include "test/mds/mock/mock_chunkserver.h"
#include "test/mds/mock/mock_topology.h"

using ::curve::mds::topology::ChunkServerState;
using ::curve::mds::topology::DISKNORMAL;
using ::curve::mds::topology::MockTopology;
using ::curve::mds::topology::OFFLINE;
using ::curve::mds::topology::ONLINE;
using ::curve::mds::topology::READWRITE;
using ::curve::mds::topology::UNINTIALIZE_ID;

using ::curve::chunkserver::CHUNK_OP_STATUS;
using ::curve::chunkserver::CHUNK_OP_STATUS_FAILURE_UNKNOWN;
using ::curve::chunkserver::CHUNK_OP_STATUS_REDIRECTED;
using ::curve::chunkserver::CHUNK_OP_TYPE;
using ::curve::chunkserver::ChunkRequest;
using ::curve::chunkserver::ChunkResponse;
using ::curve::chunkserver::MockChunkService;
using ::curve::chunkserver::MockCliService;

using ::curve::chunkserver::GetLeaderRequest2;
using ::curve::chunkserver::GetLeaderResponse2;

using ::google::protobuf::Closure;
using ::google::protobuf::RpcController;

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace curve {
namespace mds {
namespace chunkserverclient {

class TestChunkServerClient : public ::testing::Test {
 protected:
  TestChunkServerClient() {}
  void SetUp() {
    server_ = new brpc::Server();
    topo_ = std::make_shared<MockTopology>();
    auto channelPool = std::make_shared<ChannelPool>();
    client_ = std::make_shared<ChunkServerClient>(topo_, option, channelPool);

    mockCliService = new MockCliService();
    chunkService = new MockChunkService();
    ASSERT_EQ(
        server_->AddService(chunkService, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
    ASSERT_EQ(
        server_->AddService(mockCliService, brpc::SERVER_DOESNT_OWN_SERVICE),
        0);
    ASSERT_EQ(0, server_->Start("127.0.0.1", {8900, 8999}, nullptr));
    listenAddr_ = server_->listen_address();
  }
  void TearDown() {
    topo_ = nullptr;
    client_ = nullptr;

    server_->Stop(0);
    server_->Join();
    delete server_;
    server_ = nullptr;
    delete chunkService;
    chunkService = nullptr;
    delete mockCliService;
    mockCliService = nullptr;
  }

 protected:
  std::shared_ptr<MockTopology> topo_;
  std::shared_ptr<ChunkServerClient> client_;
  butil::EndPoint listenAddr_;
  brpc::Server* server_;
  MockChunkService* chunkService;
  MockCliService* mockCliService;
  ChunkServerClientOption option;
};

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotSuccess) {
  uint32_t port = listenAddr_.port;
  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS);
  EXPECT_CALL(*chunkService, DeleteChunkSnapshotOrCorrectSn(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kMdsSuccess, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotGetChunkServerFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(false)));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kMdsFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotChunkServerOFFLINE) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(OFFLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientCSOffline, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotRpcChannelInitFail) {
  uint32_t port = listenAddr_.port;
  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "", port, "", READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kRpcChannelInitFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotRpcCntlFail) {
  uint32_t port = listenAddr_.port;
  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS);
  EXPECT_CALL(*chunkService, DeleteChunkSnapshotOrCorrectSn(_, _, _, _))
      .Times(option.rpcRetryTimes)
      .WillRepeatedly(DoAll(
          SetArgPointee<2>(response),
          Invoke([this](RpcController* controller, const ChunkRequest* request,
                        ChunkResponse* response, Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(option.rpcTimeoutMs + 1000));
          })));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kRpcFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotRpcReturnFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_FAILURE_UNKNOWN);
  EXPECT_CALL(*chunkService, DeleteChunkSnapshotOrCorrectSn(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientReturnFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSnapshotReturnNotLeader) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_REDIRECTED);
  EXPECT_CALL(*chunkService, DeleteChunkSnapshotOrCorrectSn(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunkSnapshotOrCorrectSn(csId, logicalPoolId,
                                                    copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientNotLeader, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderSuccess) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  std::string leaderIp = "127.0.0.2";
  uint32_t leaderPort = port;
  std::string leaderPeer = leaderIp + ":" + std::to_string(leaderPort) + ":0";
  ChunkServerIdType leaderReturn = 0x02;
  GetLeaderResponse2 response;
  ::curve::common::Peer* peer = new ::curve::common::Peer();
  peer->set_id(leaderReturn);
  peer->set_address(leaderPeer);
  response.set_allocated_leader(peer);
  EXPECT_CALL(*mockCliService, GetLeader(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const GetLeaderRequest2* request,
                    GetLeaderResponse2* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));
  EXPECT_CALL(*topo_, FindChunkServerNotRetired(leaderIp, leaderPort))
      .WillOnce(Return(leaderReturn));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kMdsSuccess, ret);
  ASSERT_EQ(leaderReturn, leader);
}

TEST_F(TestChunkServerClient, TestGetLeaderGetChunkServerFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(false)));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kMdsFail, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderChunkServerOFFLINE) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(OFFLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kCsClientCSOffline, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderRpcChannelInitFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "", port, "", READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kRpcChannelInitFail, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderRpcCntlFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  std::string leaderIp = "127.0.0.2";
  uint32_t leaderPort = port;
  std::string leaderPeer = leaderIp + ":" + std::to_string(leaderPort) + ":0";
  ChunkServerIdType leaderReturn = 0x02;
  GetLeaderResponse2 response;
  ::curve::common::Peer* peer = new ::curve::common::Peer();
  peer->set_id(leaderReturn);
  peer->set_address(leaderPeer);
  response.set_allocated_leader(peer);
  EXPECT_CALL(*mockCliService, GetLeader(_, _, _, _))
      .WillRepeatedly(
          DoAll(SetArgPointee<2>(response),
                Invoke([this](RpcController* controller,
                              const GetLeaderRequest2* request,
                              GetLeaderResponse2* response, Closure* done) {
                  brpc::ClosureGuard doneGuard(done);
                  std::this_thread::sleep_for(
                      std::chrono::milliseconds(option.rpcTimeoutMs + 1000));
                })));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kRpcFail, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderRpcReturnLeaderPeerInvalid) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  std::string leaderPeer = "abcde";
  ChunkServerIdType leaderReturn = 0x02;
  GetLeaderResponse2 response;
  ::curve::common::Peer* peer = new ::curve::common::Peer();
  peer->set_id(leaderReturn);
  peer->set_address(leaderPeer);
  response.set_allocated_leader(peer);
  EXPECT_CALL(*mockCliService, GetLeader(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const GetLeaderRequest2* request,
                    GetLeaderResponse2* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kMdsFail, ret);
}

TEST_F(TestChunkServerClient, TestGetLeaderRpcReturnLeaderPeerNotExist) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkServerIdType leader;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  std::string leaderIp = "127.0.0.2";
  uint32_t leaderPort = port;
  std::string leaderPeer = leaderIp + ":" + std::to_string(leaderPort) + ":0";
  ChunkServerIdType leaderReturn = 0x02;
  GetLeaderResponse2 response;
  ::curve::common::Peer* peer = new ::curve::common::Peer();
  peer->set_id(leaderReturn);
  peer->set_address(leaderPeer);
  response.set_allocated_leader(peer);
  EXPECT_CALL(*mockCliService, GetLeader(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const GetLeaderRequest2* request,
                    GetLeaderResponse2* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));
  EXPECT_CALL(*topo_, FindChunkServerNotRetired(leaderIp, leaderPort))
      .WillOnce(Return(UNINTIALIZE_ID));

  int ret = client_->GetLeader(csId, logicalPoolId, copysetId, &leader);
  ASSERT_EQ(kMdsFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkSuccess) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS);
  EXPECT_CALL(*chunkService, DeleteChunk(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kMdsSuccess, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkGetChunkServerFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(false)));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kMdsFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkChunkServerOFFLINE) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(OFFLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientCSOffline, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkRpcChannelInitFail) {
  uint32_t port = listenAddr_.port;
  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "", port, "", READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kRpcChannelInitFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkRpcCntlFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_SUCCESS);
  EXPECT_CALL(*chunkService, DeleteChunk(_, _, _, _))
      .Times(option.rpcRetryTimes)
      .WillRepeatedly(DoAll(
          SetArgPointee<2>(response),
          Invoke([this](RpcController* controller, const ChunkRequest* request,
                        ChunkResponse* response, Closure* done) {
            brpc::ClosureGuard doneGuard(done);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(option.rpcTimeoutMs + 1000));
          })));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kRpcFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkRpcReturnFail) {
  uint32_t port = listenAddr_.port;

  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_FAILURE_UNKNOWN);
  EXPECT_CALL(*chunkService, DeleteChunk(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientReturnFail, ret);
}

TEST_F(TestChunkServerClient, TestDeleteChunkReturnNotLeader) {
  uint32_t port = listenAddr_.port;
  ChunkServerIdType csId = 0x01;
  LogicalPoolID logicalPoolId = 0x11;
  CopysetID copysetId = 0x21;
  ChunkID chunkId = 0x31;
  uint64_t sn = 100;

  ChunkServer chunkserver(csId, "", "", 0x101, "127.0.0.1", port, "",
                          READWRITE);
  ChunkServerState csState;
  csState.SetDiskState(DISKNORMAL);
  chunkserver.SetOnlineState(ONLINE);
  chunkserver.SetChunkServerState(csState);

  EXPECT_CALL(*topo_, GetChunkServer(csId, _))
      .WillOnce(DoAll(SetArgPointee<1>(chunkserver), Return(true)));
  ChunkResponse response;
  response.set_status(CHUNK_OP_STATUS::CHUNK_OP_STATUS_REDIRECTED);
  EXPECT_CALL(*chunkService, DeleteChunk(_, _, _, _))
      .WillOnce(DoAll(
          SetArgPointee<2>(response),
          Invoke([](RpcController* controller, const ChunkRequest* request,
                    ChunkResponse* response,
                    Closure* done) { brpc::ClosureGuard doneGuard(done); })));

  int ret = client_->DeleteChunk(csId, logicalPoolId, copysetId, chunkId, sn);
  ASSERT_EQ(kCsClientNotLeader, ret);
}

}  // namespace chunkserverclient
}  // namespace mds
}  // namespace curve
