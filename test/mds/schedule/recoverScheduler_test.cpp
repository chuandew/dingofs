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
 * Created Date: Tue Dec 25 2018
 * Author: lixiaocui
 */

#include <glog/logging.h>

#include "src/mds/common/mds_define.h"
#include "src/mds/schedule/operatorController.h"
#include "src/mds/schedule/scheduleMetrics.h"
#include "src/mds/schedule/scheduler.h"
#include "src/mds/topology/topology_id_generator.h"
#include "test/mds/mock/mock_topology.h"
#include "test/mds/schedule/common.h"
#include "test/mds/schedule/mock_topoAdapter.h"

using ::testing::_;
using ::testing::AtLeast;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

using ::curve::mds::topology::MockTopology;
using ::curve::mds::topology::TopologyIdGenerator;

namespace curve {
namespace mds {
namespace schedule {
class TestRecoverSheduler : public ::testing::Test {
 protected:
  TestRecoverSheduler() {}
  ~TestRecoverSheduler() {}

  void SetUp() override {
    auto topo = std::make_shared<MockTopology>();
    auto metric = std::make_shared<ScheduleMetrics>(topo);
    opController_ = std::make_shared<OperatorController>(2, metric);
    topoAdapter_ = std::make_shared<MockTopoAdapter>();

    ScheduleOption opt;
    opt.transferLeaderTimeLimitSec = 10;
    opt.removePeerTimeLimitSec = 100;
    opt.addPeerTimeLimitSec = 1000;
    opt.changePeerTimeLimitSec = 1000;
    opt.recoverSchedulerIntervalSec = 1;
    opt.scatterWithRangePerent = 0.2;
    opt.chunkserverFailureTolerance = 3;
    recoverScheduler_ =
        std::make_shared<RecoverScheduler>(opt, topoAdapter_, opController_);
  }
  void TearDown() override {
    opController_ = nullptr;
    topoAdapter_ = nullptr;
    recoverScheduler_ = nullptr;
  }

 protected:
  std::shared_ptr<MockTopoAdapter> topoAdapter_;
  std::shared_ptr<OperatorController> opController_;
  std::shared_ptr<RecoverScheduler> recoverScheduler_;
};

TEST_F(TestRecoverSheduler, test_copySet_already_has_operator) {
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillOnce(Return(std::vector<CopySetInfo>({GetCopySetInfoForTest()})));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillOnce(Return(std::vector<ChunkServerInfo>{}));
  CopySetKey copySetKey;
  copySetKey.first = 1;
  copySetKey.second = 1;
  Operator testOperator(1, copySetKey, OperatorPriority::NormalPriority,
                        steady_clock::now(), std::make_shared<AddPeer>(1));
  ASSERT_TRUE(opController_->AddOperator(testOperator));
  recoverScheduler_->Schedule();
  ASSERT_EQ(1, opController_->GetOperators().size());
}

TEST_F(TestRecoverSheduler, test_copySet_has_configChangeInfo) {
  auto testCopySetInfo = GetCopySetInfoForTest();
  testCopySetInfo.candidatePeerInfo = PeerInfo(1, 1, 1, "", 9000);
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillOnce(Return(std::vector<ChunkServerInfo>{}));
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillOnce(Return(std::vector<CopySetInfo>({testCopySetInfo})));
  recoverScheduler_->Schedule();
  ASSERT_EQ(0, opController_->GetOperators().size());
}

TEST_F(TestRecoverSheduler, test_chunkServer_cannot_get) {
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillOnce(Return(std::vector<ChunkServerInfo>{}));
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillOnce(Return(std::vector<CopySetInfo>({GetCopySetInfoForTest()})));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(_, _))
      .Times(3)
      .WillRepeatedly(Return(false));
  recoverScheduler_->Schedule();
  ASSERT_EQ(0, opController_->GetOperators().size());
}

TEST_F(TestRecoverSheduler, test_server_has_more_offline_chunkserver) {
  auto testCopySetInfo = GetCopySetInfoForTest();
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillRepeatedly(Return(std::vector<CopySetInfo>({testCopySetInfo})));
  ChunkServerInfo csInfo1(testCopySetInfo.peers[0], OnlineState::OFFLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo2(testCopySetInfo.peers[1], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo3(testCopySetInfo.peers[2], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  PeerInfo peer4(4, 1, 1, "192.168.10.1", 9001);
  PeerInfo peer5(5, 1, 1, "192.168.10.1", 9002);
  ChunkServerInfo csInfo4(peer4, OnlineState::OFFLINE, DiskState::DISKNORMAL,
                          ChunkServerStatus::READWRITE, 2, 100, 100,
                          ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo5(peer5, OnlineState::UNSTABLE, DiskState::DISKNORMAL,
                          ChunkServerStatus::READWRITE, 2, 100, 100,
                          ChunkServerStatisticInfo{});
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillOnce(Return(std::vector<ChunkServerInfo>{csInfo1, csInfo2, csInfo3,
                                                    csInfo4, csInfo5}));

  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo1.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo1), Return(true)));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo2.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo2), Return(true)));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo3.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo3), Return(true)));
  recoverScheduler_->Schedule();
  ASSERT_EQ(0, opController_->GetOperators().size());
}

TEST_F(TestRecoverSheduler,
       test_server_has_more_offline_and_retired_chunkserver) {
  auto testCopySetInfo = GetCopySetInfoForTest();
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillRepeatedly(Return(std::vector<CopySetInfo>({testCopySetInfo})));
  ChunkServerInfo csInfo1(testCopySetInfo.peers[0], OnlineState::OFFLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::PENDDING, 2,
                          100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo2(testCopySetInfo.peers[1], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo3(testCopySetInfo.peers[2], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  PeerInfo peer4(4, 1, 1, "192.168.10.1", 9001);
  PeerInfo peer5(5, 1, 1, "192.168.10.1", 9002);
  ChunkServerInfo csInfo4(peer4, OnlineState::OFFLINE, DiskState::DISKNORMAL,
                          ChunkServerStatus::READWRITE, 2, 100, 100,
                          ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo5(peer5, OnlineState::OFFLINE, DiskState::DISKNORMAL,
                          ChunkServerStatus::READWRITE, 2, 100, 100,
                          ChunkServerStatisticInfo{});
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillOnce(Return(std::vector<ChunkServerInfo>{csInfo1, csInfo2, csInfo3,
                                                    csInfo4, csInfo5}));

  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo1.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo1), Return(true)));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo2.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo2), Return(true)));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(csInfo3.info.id, _))
      .WillOnce(DoAll(SetArgPointee<1>(csInfo3), Return(true)));
  EXPECT_CALL(*topoAdapter_, GetStandardReplicaNumInLogicalPool(_))
      .WillOnce(Return(2));
  recoverScheduler_->Schedule();
  Operator op;
  ASSERT_TRUE(opController_->GetOperatorById(testCopySetInfo.id, &op));
  ASSERT_TRUE(dynamic_cast<RemovePeer*>(op.step.get()) != nullptr);
  ASSERT_EQ(std::chrono::seconds(100), op.timeLimit);
}

TEST_F(TestRecoverSheduler, test_all_chunkServer_online_offline) {
  auto testCopySetInfo = GetCopySetInfoForTest();
  EXPECT_CALL(*topoAdapter_, GetCopySetInfos())
      .WillRepeatedly(Return(std::vector<CopySetInfo>({testCopySetInfo})));
  EXPECT_CALL(*topoAdapter_, GetChunkServerInfos())
      .WillRepeatedly(Return(std::vector<ChunkServerInfo>{}));
  ChunkServerInfo csInfo1(testCopySetInfo.peers[0], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo2(testCopySetInfo.peers[1], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  ChunkServerInfo csInfo3(testCopySetInfo.peers[2], OnlineState::ONLINE,
                          DiskState::DISKNORMAL, ChunkServerStatus::READWRITE,
                          2, 100, 100, ChunkServerStatisticInfo{});
  PeerInfo peer4(4, 4, 4, "192.168.10.4", 9000);
  ChunkServerInfo csInfo4(peer4, OnlineState::ONLINE, DiskState::DISKNORMAL,
                          ChunkServerStatus::READWRITE, 2, 100, 100,
                          ChunkServerStatisticInfo{});
  ChunkServerIdType id1 = 1;
  ChunkServerIdType id2 = 2;
  ChunkServerIdType id3 = 3;
  Operator op;
  EXPECT_CALL(*topoAdapter_, GetAvgScatterWidthInLogicalPool(_))
      .WillRepeatedly(Return(90));
  {
    // 1. 所有chunkserveronline
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id1, _))
        .WillOnce(DoAll(SetArgPointee<1>(csInfo1), Return(true)));
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id2, _))
        .WillRepeatedly(DoAll(SetArgPointee<1>(csInfo2), Return(true)));
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id3, _))
        .WillRepeatedly(DoAll(SetArgPointee<1>(csInfo3), Return(true)));
    recoverScheduler_->Schedule();
    ASSERT_EQ(0, opController_->GetOperators().size());
  }

  {
    // 2. 副本数量大于标准，leader挂掉
    csInfo1.state = OnlineState::OFFLINE;
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id1, _))
        .WillOnce(DoAll(SetArgPointee<1>(csInfo1), Return(true)));
    EXPECT_CALL(*topoAdapter_, GetStandardReplicaNumInLogicalPool(_))
        .Times(2)
        .WillRepeatedly(Return(2));
    recoverScheduler_->Schedule();
    ASSERT_TRUE(opController_->GetOperatorById(testCopySetInfo.id, &op));
    ASSERT_TRUE(dynamic_cast<RemovePeer*>(op.step.get()) != nullptr);
    ASSERT_EQ(std::chrono::seconds(100), op.timeLimit);
  }

  {
    // 3. 副本数量大于标准，follower挂掉
    opController_->RemoveOperator(op.copysetID);
    csInfo1.state = OnlineState::ONLINE;
    csInfo2.state = OnlineState::OFFLINE;
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id1, _))
        .WillRepeatedly(DoAll(SetArgPointee<1>(csInfo1), Return(true)));
    EXPECT_CALL(*topoAdapter_, GetChunkServerInfo(id2, _))
        .WillRepeatedly(DoAll(SetArgPointee<1>(csInfo2), Return(true)));
    recoverScheduler_->Schedule();
    ASSERT_TRUE(opController_->GetOperatorById(testCopySetInfo.id, &op));
    ASSERT_TRUE(dynamic_cast<RemovePeer*>(op.step.get()) != nullptr);
    ASSERT_EQ(std::chrono::seconds(100), op.timeLimit);
  }

  {
    // 4. 副本数目等于标准， follower挂掉
    opController_->RemoveOperator(op.copysetID);
    EXPECT_CALL(*topoAdapter_, GetStandardReplicaNumInLogicalPool(_))
        .WillRepeatedly(Return(3));
    std::vector<ChunkServerInfo> chunkserverList(
        {csInfo1, csInfo2, csInfo3, csInfo4});
    EXPECT_CALL(*topoAdapter_, GetChunkServersInLogicalPool(_))
        .WillOnce(Return(chunkserverList));
    EXPECT_CALL(*topoAdapter_, GetStandardZoneNumInLogicalPool(_))
        .WillOnce(Return(3));
    std::map<ChunkServerIdType, int> map1{{3, 1}};
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(1, _))
        .WillOnce(SetArgPointee<1>(map1));
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(2, _))
        .WillOnce(SetArgPointee<1>(map1));
    std::map<ChunkServerIdType, int> map3{{1, 1}};
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(3, _))
        .WillOnce(SetArgPointee<1>(map3));
    std::map<ChunkServerIdType, int> map4;
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(4, _))
        .WillOnce(SetArgPointee<1>(map4));
    EXPECT_CALL(*topoAdapter_, CreateCopySetAtChunkServer(_, _))
        .WillOnce(Return(true));
    recoverScheduler_->Schedule();
    ASSERT_TRUE(opController_->GetOperatorById(testCopySetInfo.id, &op));
    ASSERT_TRUE(dynamic_cast<ChangePeer*>(op.step.get()) != nullptr);
    ASSERT_EQ(4, op.step.get()->GetTargetPeer());
    ASSERT_EQ(std::chrono::seconds(1000), op.timeLimit);
  }

  {
    // 5. 选不出替换chunkserver
    opController_->RemoveOperator(op.copysetID);
    EXPECT_CALL(*topoAdapter_, GetChunkServersInLogicalPool(_))
        .WillOnce(Return(std::vector<ChunkServerInfo>{}));
    recoverScheduler_->Schedule();
    ASSERT_EQ(0, opController_->GetOperators().size());
  }

  {
    // 6. 在chunkserver上创建copyset失败
    EXPECT_CALL(*topoAdapter_, GetStandardReplicaNumInLogicalPool(_))
        .WillRepeatedly(Return(3));
    std::vector<ChunkServerInfo> chunkserverList(
        {csInfo1, csInfo2, csInfo3, csInfo4});
    EXPECT_CALL(*topoAdapter_, GetChunkServersInLogicalPool(_))
        .WillOnce(Return(chunkserverList));
    EXPECT_CALL(*topoAdapter_, GetStandardZoneNumInLogicalPool(_))
        .WillOnce(Return(3));
    std::map<ChunkServerIdType, int> map1{{3, 1}};
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(1, _))
        .WillOnce(SetArgPointee<1>(map1));
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(2, _))
        .WillOnce(SetArgPointee<1>(map1));
    std::map<ChunkServerIdType, int> map3{{1, 1}};
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(3, _))
        .WillOnce(SetArgPointee<1>(map3));
    std::map<ChunkServerIdType, int> map4;
    EXPECT_CALL(*topoAdapter_, GetChunkServerScatterMap(4, _))
        .WillOnce(SetArgPointee<1>(map4));
    EXPECT_CALL(*topoAdapter_, CreateCopySetAtChunkServer(_, _))
        .WillOnce(Return(false));
    recoverScheduler_->Schedule();
    ASSERT_EQ(0, opController_->GetOperators().size());
  }
}
}  // namespace schedule
}  // namespace mds
}  // namespace curve
