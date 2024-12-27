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
 * Created Date: 2021-10-30
 * Author: chengyi01
 */

#ifndef DINGOFS_SRC_TOOLS_QUERY_DINGOFS_COPYSET_QUERY_H_
#define DINGOFS_SRC_TOOLS_QUERY_DINGOFS_COPYSET_QUERY_H_

#include <brpc/channel.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "dingofs/proto/copyset.pb.h"
#include "dingofs/proto/mds.pb.h"
#include "dingofs/proto/topology.pb.h"
#include "dingofs/src/mds/topology/deal_peerid.h"
#include "dingofs/src/tools/copyset/dingofs_copyset_status.h"
#include "dingofs/src/tools/dingofs_tool.h"
#include "dingofs/src/tools/dingofs_tool_define.h"
#include "dingofs/src/utils/string_util.h"

namespace dingofs {
namespace tools {
namespace query {

using InfoType = dingofs::mds::topology::CopysetValue;
using StatusType = dingofs::metaserver::copyset::CopysetStatusResponse;
using StatusRequestType = dingofs::metaserver::copyset::CopysetsStatusRequest;

class CopysetQueryTool
    : public CurvefsToolRpc<dingofs::mds::topology::GetCopysetsInfoRequest,
                            dingofs::mds::topology::GetCopysetsInfoResponse,
                            dingofs::mds::topology::TopologyService_Stub> {
 public:
  explicit CopysetQueryTool(const std::string& cmd = kCopysetQueryCmd,
                            bool show = true)
      : CurvefsToolRpc(cmd) {
    show_ = show;
  }
  void PrintHelp() override;
  int Init() override;

  std::map<uint64_t, std::vector<InfoType>> GetKey2Info() { return key2Info_; }
  std::map<uint64_t, std::vector<StatusType>> GetKey2Status() {
    return key2Status_;
  }
  std::vector<uint64_t> GetKey_() { return copysetKeys_; }

 protected:
  void AddUpdateFlags() override;
  bool AfterSendRequestToHost(const std::string& host) override;
  bool GetCopysetStatus();
  bool CheckRequiredFlagDefault() override;

 protected:
  // key = (poolId << 32) | copysetId
  std::map<uint64_t, std::vector<InfoType>> key2Info_;
  std::map<std::string, std::queue<StatusRequestType>> addr2Request_;  // detail
  std::map<uint64_t, std::vector<StatusType>> key2Status_;             // detail
  std::vector<uint64_t> copysetKeys_;
};

}  // namespace query
}  // namespace tools
}  // namespace dingofs

#endif  // DINGOFS_SRC_TOOLS_QUERY_DINGOFS_COPYSET_QUERY_H_