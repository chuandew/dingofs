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
 * Project: curve
 * Created Date: 2021-09-03
 * Author: chengyi01
 */

#ifndef CURVEFS_SRC_TOOLS_CREATE_CURVEFS_CREATE_TOPOLOGY_TOOL_H_
#define CURVEFS_SRC_TOOLS_CREATE_CURVEFS_CREATE_TOPOLOGY_TOOL_H_

#include <brpc/channel.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <json/json.h>

#include <list>
#include <string>
#include <vector>

#include "curvefs/proto/topology.pb.h"
#include "curvefs/src/mds/common/mds_define.h"
#include "curvefs/src/tools/curvefs_tool.h"
#include "curvefs/src/tools/curvefs_tool_define.h"
#include "curvefs/src/utils/configuration.h"

namespace curvefs {
namespace mds {
namespace topology {
struct Server {
  std::string name;
  std::string internalIp;
  uint32_t internalPort;
  std::string externalIp;
  uint32_t externalPort;
  std::string zoneName;
  std::string poolName;
};

struct Pool {
  std::string name;
  uint32_t replicasNum;
  uint64_t copysetNum;
  uint32_t zoneNum;
};

struct Zone {
  std::string name;
  std::string poolName;
};

class CurvefsBuildTopologyTool : public curvefs::tools::CurvefsTool {
 public:
  CurvefsBuildTopologyTool()
      : curvefs::tools::CurvefsTool(
            std::string(curvefs::tools::kCreateTopologyCmd),
            std::string(curvefs::tools::kProgrameName)) {}

  ~CurvefsBuildTopologyTool() override = default;

  int Init() override;

  int InitTopoData();

  int HandleBuildCluster();

  int GetMaxTry() { return mdsAddressStr_.size(); }

  int TryAnotherMdsAddress();

  void PrintHelp() override {
    std::cout << "Example :" << std::endl;
    std::cout << programe_ << " " << command_ << " "
              << curvefs::tools::kConfPathHelp << std::endl;
  }

  int RunCommand() override;

  int Run() override {
    if (Init() < 0) {
      LOG(ERROR) << "CurvefsBuildTopologyTool init error.";
      return kRetCodeCommonErr;
    }
    int ret = InitTopoData();
    if (ret < 0) {
      LOG(ERROR) << "Init topo json data error.";
      return ret;
    }
    ret = RunCommand();
    return ret;
  }

 public:
  // for unit test
  const std::list<Server>& GetServerDatas() { return serverDatas_; }

  const std::list<Zone>& GetZoneDatas() { return zoneDatas_; }

  const std::list<Pool>& GetPoolDatas() { return poolDatas_; }

 private:
  int ReadClusterMap();
  int InitServerZoneData();
  int InitPoolData();
  int ScanCluster();
  int ScanPool();
  int RemovePoolsNotInNewTopo();
  int RemoveZonesNotInNewTopo();
  int RemoveServersNotInNewTopo();
  int CreatePool();
  int CreateZone();
  int CreateServer();

  int DealFailedRet(int ret, std::string operation);

  int ListPool(std::list<PoolInfo>* pool_infos);

  int GetZonesInPool(PoolIdType poolid, std::list<ZoneInfo>* zone_infos);

  int GetServersInZone(ZoneIdType zoneid, std::list<ServerInfo>* server_infos);

  std::list<Server> serverDatas_;
  std::list<Zone> zoneDatas_;
  std::list<Pool> poolDatas_;

  std::list<ServerIdType> serverToDel_;
  std::list<ZoneIdType> zoneToDel_;
  std::list<PoolIdType> poolToDel_;

  std::vector<std::string> mdsAddressStr_;
  int mdsAddressIndex_;
  brpc::Channel channel_;
  Json::Value clusterMap_;
};

}  // namespace topology
}  // namespace mds
}  // namespace curvefs

#endif  // CURVEFS_SRC_TOOLS_CREATE_CURVEFS_CREATE_TOPOLOGY_TOOL_H_
