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
 * Created Date: Thur Apr 24th 2019
 * Author: lixiaocui
 */

#include "src/mds/nameserver2/idgenerator/chunk_id_generator.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>  //NOLINT
#include <thread>  //NOLINT

#include "src/mds/nameserver2/helper/namespace_helper.h"
#include "test/mds/mock/mock_etcdclient.h"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

namespace curve {
namespace mds {
class TestChunkIdGenerator : public ::testing::Test {
 protected:
  TestChunkIdGenerator() {}
  ~TestChunkIdGenerator() {}

  void SetUp() override {
    client_ = std::make_shared<MockEtcdClient>();
    chunkIdGen_ = std::make_shared<ChunkIDGeneratorImp>(client_);
  }

  void TearDown() override {
    client_ = nullptr;
    chunkIdGen_ = nullptr;
  }

 protected:
  std::shared_ptr<MockEtcdClient> client_;
  std::shared_ptr<ChunkIDGeneratorImp> chunkIdGen_;
};

TEST_F(TestChunkIdGenerator, test_all) {
  uint64_t alloc1 = CHUNKINITIALIZE + CHUNKBUNDLEALLOCATED;
  uint64_t alloc2 = alloc1 + CHUNKBUNDLEALLOCATED;
  std::string strAlloc1 = NameSpaceStorageCodec::EncodeID(alloc1);
  EXPECT_CALL(*client_, Get(CHUNKSTOREKEY, _))
      .WillOnce(Return(EtcdErrCode::EtcdKeyNotExist))
      .WillOnce(
          DoAll(SetArgPointee<1>(strAlloc1), Return(EtcdErrCode::EtcdOK)));
  EXPECT_CALL(*client_, CompareAndSwap(CHUNKSTOREKEY, "",
                                       NameSpaceStorageCodec::EncodeID(alloc1)))
      .WillOnce(Return(EtcdErrCode::EtcdOK));
  EXPECT_CALL(*client_, CompareAndSwap(CHUNKSTOREKEY, strAlloc1,
                                       NameSpaceStorageCodec::EncodeID(alloc2)))
      .WillOnce(Return(EtcdErrCode::EtcdOK));

  uint64_t end = 2 * CHUNKBUNDLEALLOCATED;
  InodeID res;
  for (int i = CHUNKINITIALIZE + 1; i <= end; i++) {
    ASSERT_TRUE(chunkIdGen_->GenChunkID(&res));
    ASSERT_EQ(i, res);
  }
}
}  // namespace mds
}  // namespace curve
