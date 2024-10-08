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
 * Created Date: 18-12-20
 * Author: wudemiao
 */

#include "src/chunkserver/conf_epoch_file.h"

#include <gtest/gtest.h>

#include <memory>

#include "src/chunkserver/copyset_node.h"
#include "src/fs/fs_common.h"
#include "test/fs/mock_local_filesystem.h"

namespace curve {
namespace chunkserver {

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::DoAll;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::Matcher;
using ::testing::Return;
using ::testing::SaveArgPointee;
using ::testing::SetArgPointee;
using ::testing::SetArgReferee;
using ::testing::SetArrayArgument;

using curve::fs::FileSystemType;
using curve::fs::MockLocalFileSystem;

TEST(ConfEpochFileTest, load_save) {
  LogicPoolID logicPoolID = 123;
  CopysetID copysetID = 1345;
  std::string path = kCurveConfEpochFilename;
  uint64_t epoch = 0;
  std::string rmCmd("rm -f ");
  rmCmd += kCurveConfEpochFilename;

  // normal load/save
  {
    auto fs = LocalFsFactory::CreateFs(FileSystemType::EXT4, "");
    ConfEpochFile confEpochFile(fs);
    ASSERT_EQ(0, confEpochFile.Save(path, logicPoolID, copysetID, epoch));

    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    ASSERT_EQ(0, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                    &loadEpoch));
    ASSERT_EQ(logicPoolID, loadLogicPoolID);
    ASSERT_EQ(copysetID, loadCopysetID);
    ASSERT_EQ(epoch, loadEpoch);

    ::system(rmCmd.c_str());
  }

  // load: open failed
  {
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(-1));
    ASSERT_EQ(-1, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                     &loadEpoch));
  }
  // load: open success, read failed
  {
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Read(_, _, _, _)).Times(1).WillOnce(Return(-1));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    ASSERT_EQ(-1, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                     &loadEpoch));
  }
  // load: open success, read success, decode success, crc32c right
  {
    const char* json =
        "{\"logicPoolId\":123,\"copysetId\":1345,\"epoch\":"
        "0,\"checksum\":599727352}";  // NOLINT
    std::string jsonStr(json);
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Read(_, _, _, _))
        .Times(1)
        .WillOnce(DoAll(SetArrayArgument<1>(json, json + jsonStr.size()),
                        Return(jsonStr.size())));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    ASSERT_EQ(0, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                    &loadEpoch));
  }
  // load: open success, read success, decode failed, crc32c right
  {
    const char* json = "{\"logicPoolId";
    std::string jsonStr(json);
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Read(_, _, _, _))
        .Times(1)
        .WillOnce(DoAll(SetArrayArgument<1>(json, json + jsonStr.size()),
                        Return(jsonStr.size())));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    ASSERT_EQ(-1, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                     &loadEpoch));
  }
  // load: open success, read success, decode success, crc32c not right
  {
    const char* json =
        "{\"logicPoolId\":123,\"copysetId\":1345,\"epoch\":"
        "0,\"checksum\":123}";  // NOLINT
    std::string jsonStr(json);
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID;
    CopysetID loadCopysetID;
    uint64_t loadEpoch;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Read(_, _, _, _))
        .Times(1)
        .WillOnce(DoAll(SetArrayArgument<1>(json, json + jsonStr.size()),
                        Return(jsonStr.size())));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    ASSERT_EQ(-1, confEpochFile.Load(path, &loadLogicPoolID, &loadCopysetID,
                                     &loadEpoch));
  }
  // save: open failed
  {
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID = 0;
    CopysetID loadCopysetID = 0;
    uint64_t loadEpoch = 0;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(-1));
    ASSERT_EQ(-1, confEpochFile.Save(path, loadLogicPoolID, loadCopysetID,
                                     loadEpoch));
  }
  // save: open success, write failed
  {
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    LogicPoolID loadLogicPoolID = 0;
    CopysetID loadCopysetID = 0;
    uint64_t loadEpoch = 0;
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Write(_, Matcher<const char*>(_), _, _))
        .Times(1)
        .WillOnce(Return(-1));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    ASSERT_EQ(-1, confEpochFile.Save(path, loadLogicPoolID, loadCopysetID,
                                     loadEpoch));
  }
  // save: open success, write success, fsync failed
  {
    const char* json =
        "{\"logicPoolId\":123,\"copysetId\":1345,\"epoch\":"
        "0,\"checksum\":599727352}";  // NOLINT
    std::string jsonStr(json);
    std::shared_ptr<MockLocalFileSystem> fs =
        std::make_shared<MockLocalFileSystem>();
    ConfEpochFile confEpochFile(fs);
    EXPECT_CALL(*fs, Open(_, _)).Times(1).WillOnce(Return(10));
    EXPECT_CALL(*fs, Write(_, Matcher<const char*>(_), _, _))
        .Times(1)
        .WillOnce(Return(jsonStr.size()));
    EXPECT_CALL(*fs, Close(_)).Times(1).WillOnce(Return(0));
    EXPECT_CALL(*fs, Fsync(_)).Times(1).WillOnce(Return(-1));
    ASSERT_EQ(-1, confEpochFile.Save(path, logicPoolID, copysetID, epoch));
  }
}

}  // namespace chunkserver
}  // namespace curve
