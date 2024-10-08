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
 * Created Date: Sat Jan 05 2019
 * Author: lixiaocui
 */

#ifndef TEST_MDS_HEARTBEAT_COMMON_H_
#define TEST_MDS_HEARTBEAT_COMMON_H_

#include "proto/heartbeat.pb.h"

namespace curve {
namespace mds {
namespace heartbeat {
ChunkServerHeartbeatRequest GetChunkServerHeartbeatRequestForTest();
}  // namespace heartbeat
}  // namespace mds
}  // namespace curve
#endif  // TEST_MDS_HEARTBEAT_COMMON_H_
