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
 * Created Date: 2019-11-28
 * Author: charisu
 */

#ifndef SRC_TOOLS_CHUNKSERVER_CLIENT_H_
#define SRC_TOOLS_CHUNKSERVER_CLIENT_H_

#include <braft/builtin_service.pb.h>
#include <brpc/channel.h>
#include <gflags/gflags.h>

#include <iostream>
#include <string>

#include "proto/chunk.pb.h"
#include "proto/copyset.pb.h"
#include "src/tools/curve_tool_define.h"

using curve::chunkserver::CHUNK_OP_STATUS;
using curve::chunkserver::COPYSET_OP_STATUS;
using curve::chunkserver::CopysetStatusRequest;
using curve::chunkserver::CopysetStatusResponse;
using curve::chunkserver::GetChunkHashRequest;
using curve::chunkserver::GetChunkHashResponse;

namespace curve {
namespace tool {

struct Chunk {
  Chunk(uint32_t poolId, uint32_t csId, uint64_t chunkId2)
      : logicPoolId(poolId), copysetId(csId), chunkId(chunkId2) {}
  uint32_t logicPoolId;
  uint32_t copysetId;
  uint64_t chunkId;
};

std::ostream& operator<<(std::ostream& os, const Chunk& chunk);

class ChunkServerClient {
 public:
  virtual ~ChunkServerClient() = default;
  /**
   *  @brief 初始化channel，对一个地址，初始化一次就好
   *  @param csAddr chunkserver地址
   *  @return 成功返回0，失败返回-1
   */
  virtual int Init(const std::string& csAddr);

  /**
   *  @brief 调用braft的RaftStat接口获取复制组的详细信息，放到iobuf里面
   *  @param iobuf 复制组详细信息，返回值为0时有效
   *  @return 成功返回0，失败返回-1
   */
  virtual int GetRaftStatus(butil::IOBuf* iobuf);

  /**
   *  @brief 检查chunkserver是否在线，只检查controller，不检查response
   *  @return 在线返回true，不在线返回false
   */
  virtual bool CheckChunkServerOnline();

  /**
  *  @brief 调用chunkserver的GetCopysetStatus接口
  &  @param request 查询copyset的request
  *  @param response 返回的response，里面有复制组详细信息，返回值为0时有效
  *  @return 成功返回0，失败返回-1
  */
  virtual int GetCopysetStatus(const CopysetStatusRequest& request,
                               CopysetStatusResponse* response);

  /**
  *  @brief 从chunkserver获取chunk的hash值
  &  @param chunk 要查询的chunk
  *  @param[out] chunkHash chunk的hash值，返回值为0时有效
  *  @return 成功返回0，失败返回-1
  */
  virtual int GetChunkHash(const Chunk& chunk, std::string* chunkHash);

 private:
  brpc::Channel channel_;
  std::string csAddr_;
};

}  // namespace tool
}  // namespace curve

#endif  // SRC_TOOLS_CHUNKSERVER_CLIENT_H_
