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
 * Created Date: 2021-09-12
 * Author: chenwei
 */

#ifndef DINGOFS_SRC_METASERVER_HEARTBEAT_H_
#define DINGOFS_SRC_METASERVER_HEARTBEAT_H_

#include <braft/node.h>  // NodeImpl
#include <braft/node_manager.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "dingofs/proto/heartbeat.pb.h"
#include "dingofs/src/metaserver/common/types.h"
#include "dingofs/src/metaserver/copyset/copyset_node_manager.h"
#include "dingofs/src/utils/concurrent/concurrent.h"
#include "dingofs/src/utils/wait_interval.h"

using ::dingofs::utils::Thread;
using ::dingofs::metaserver::copyset::CopysetNode;

namespace dingofs {
namespace metaserver {

using ::dingofs::fs::LocalFileSystem;
using HeartbeatRequest = dingofs::mds::heartbeat::MetaServerHeartbeatRequest;
using HeartbeatResponse = dingofs::mds::heartbeat::MetaServerHeartbeatResponse;
using MetaServerSpaceStatus = dingofs::mds::heartbeat::MetaServerSpaceStatus;
using ::dingofs::mds::heartbeat::CopySetConf;
using TaskStatus = butil::Status;
using CopysetNodePtr = std::shared_ptr<CopysetNode>;
using dingofs::common::Peer;
using dingofs::metaserver::copyset::CopysetNodeManager;
using PeerId = braft::PeerId;

class ResourceCollector;

/**
 * heartbeat subsystem option
 */
struct HeartbeatOptions {
  MetaServerID metaserverId;
  std::string metaserverToken;
  std::string storeUri;
  std::string mdsListenAddr;
  std::string ip;
  uint32_t port;
  uint32_t intervalSec;
  uint32_t timeout;
  CopysetNodeManager* copysetNodeManager;
  ResourceCollector* resourceCollector;
  std::shared_ptr<LocalFileSystem> fs;
};

class HeartbeatTaskExecutor;

/**
 * heartbeat subsystem
 */
class Heartbeat {
 public:
  Heartbeat() = default;
  ~Heartbeat() = default;

  /**
   * @brief init heartbeat subsystem
   * @param[in] options
   * @return 0:success; not 0: fail
   */
  int Init(const HeartbeatOptions& options);

  /**
   * @brief clean heartbeat subsystem
   * @return 0:success; not 0: fail
   */
  int Fini();

  /**
   * @brief run heartbeat subsystem
   * @return 0:success; not 0: fail
   */
  int Run();

 private:
  /**
   * @brief stop heartbeat subsystem
   * @return 0:success; not 0: fail
   */
  int Stop();

  /*
   * heartbeat work thread
   */
  void HeartbeatWorker();

  void BuildCopysetInfo(dingofs::mds::heartbeat::CopySetInfo* info,
                        CopysetNode* copyset);

  int BuildRequest(HeartbeatRequest* request);

  int SendHeartbeat(const HeartbeatRequest& request,
                    HeartbeatResponse* response);

  /*
   * print HeartbeatRequest to log
   */
  void DumpHeartbeatRequest(const HeartbeatRequest& request);

  /*
   * print HeartbeatResponse to log
   */
  void DumpHeartbeatResponse(const HeartbeatResponse& response);

  bool GetMetaserverSpaceStatus(MetaServerSpaceStatus* status,
                                uint64_t ncopysets);

 private:
  friend class HeartbeatTest;

  Thread hbThread_;

  std::atomic<bool> toStop_;

  ::dingofs::utils::WaitInterval waitInterval_;

  CopysetNodeManager* copysetMan_;

  // metaserver store path
  std::string storePath_;

  HeartbeatOptions options_;

  // MDS addr list
  std::vector<std::string> mdsEps_;

  // index of current service mds
  int inServiceIndex_;

  // MetaServer addr
  butil::EndPoint msEp_;

  // heartbeat subsystem init time, use unix time
  uint64_t startUpTime_;

  std::unique_ptr<HeartbeatTaskExecutor> taskExecutor_;
};

// execute tasks from heartbeat response
class HeartbeatTaskExecutor {
 public:
  HeartbeatTaskExecutor(CopysetNodeManager* mgr,
                        const butil::EndPoint& endpoint);

  void ExecTasks(const HeartbeatResponse& response);

 private:
  void ExecOneTask(const CopySetConf& conf);

  void DoTransferLeader(CopysetNode* node, const CopySetConf& conf);
  void DoAddPeer(CopysetNode* node, const CopySetConf& conf);
  void DoRemovePeer(CopysetNode* node, const CopySetConf& conf);
  void DoChangePeer(CopysetNode* node, const CopySetConf& conf);
  void DoPurgeCopyset(PoolId poolid, CopysetId copysetid);

  bool NeedPurge(const CopySetConf& conf);

  CopysetNodeManager* copysetMgr_;
  butil::EndPoint ep_;
};

}  // namespace metaserver
}  // namespace dingofs

#endif  // DINGOFS_SRC_METASERVER_HEARTBEAT_H_