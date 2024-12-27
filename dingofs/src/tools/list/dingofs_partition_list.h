/*
 *  Copyright (c) 2022 NetEase Inc.
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
 * Created Date: 2022-04-28
 * Author: chengyi01
 */

#ifndef DINGOFS_SRC_TOOLS_LIST_DINGOFS_PARTITION_LIST_H_
#define DINGOFS_SRC_TOOLS_LIST_DINGOFS_PARTITION_LIST_H_

#include <brpc/channel.h>
#include <gflags/gflags.h>

#include <string>
#include <unordered_map>

#include "dingofs/proto/common.pb.h"
#include "dingofs/proto/topology.pb.h"
#include "dingofs/src/tools/dingofs_tool.h"
#include "dingofs/src/tools/dingofs_tool_define.h"

namespace dingofs {
namespace tools {
namespace list {

using PartitionInfoList =
    google::protobuf::RepeatedPtrField<common::PartitionInfo>;

class PartitionListTool
    : public CurvefsToolRpc<dingofs::mds::topology::ListPartitionRequest,
                            dingofs::mds::topology::ListPartitionResponse,
                            dingofs::mds::topology::TopologyService_Stub> {
 public:
  explicit PartitionListTool(const std::string& cmd = kPartitionListCmd,
                             bool show = true)
      : CurvefsToolRpc(cmd) {
    show_ = show;
  }
  void PrintHelp() override;
  int Init() override;

  std::unordered_map<uint32_t, PartitionInfoList> GetFsId2PartitionInfoList() {
    return fsId2PartitionList_;
  }

 protected:
  void AddUpdateFlags() override;
  bool AfterSendRequestToHost(const std::string& host) override;

 protected:
  std::unordered_map<uint32_t, PartitionInfoList> fsId2PartitionList_;
};
}  // namespace list
}  // namespace tools
}  // namespace dingofs

#endif  // DINGOFS_SRC_TOOLS_LIST_DINGOFS_PARTITION_LIST_H_