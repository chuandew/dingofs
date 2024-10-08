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
 * Created Date: Saturday March 30th 2019
 * Author: yangyaokai
 */

#ifndef TEST_CHUNKSERVER_CLONE_MOCK_CLONE_MANAGER_H_
#define TEST_CHUNKSERVER_CLONE_MOCK_CLONE_MANAGER_H_

#include <gmock/gmock.h>

#include <memory>
#include <string>

#include "src/chunkserver/clone_manager.h"

namespace curve {
namespace chunkserver {

class MockCloneManager : public CloneManager {
 public:
  MockCloneManager() = default;
  ~MockCloneManager() = default;
  MOCK_METHOD1(Init, int(const CloneOptions&));
  MOCK_METHOD0(Run, int());
  MOCK_METHOD0(Fini, int());
  MOCK_METHOD2(GenerateCloneTask,
               std::shared_ptr<CloneTask>(std::shared_ptr<ReadChunkRequest>,
                                          ::google::protobuf::Closure*));
  MOCK_METHOD1(IssueCloneTask, bool(std::shared_ptr<CloneTask>));
};

}  // namespace chunkserver
}  // namespace curve

#endif  // TEST_CHUNKSERVER_CLONE_MOCK_CLONE_MANAGER_H_
