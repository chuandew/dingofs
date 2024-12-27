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
 * Created Date: 2021-08-24
 * Author: wanghai01
 */

#ifndef DINGOFS_SRC_MDS_TOPOLOGY_TOPOLOGY_MANAGER_H_
#define DINGOFS_SRC_MDS_TOPOLOGY_TOPOLOGY_MANAGER_H_

#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "dingofs/proto/topology.pb.h"
#include "dingofs/src/mds/metaserverclient/metaserver_client.h"
#include "dingofs/src/mds/topology/topology.h"
#include "dingofs/src/utils/concurrent/name_lock.h"

namespace dingofs {
namespace mds {
namespace topology {

using dingofs::mds::MetaserverClient;
using dingofs::utils::NameLock;
using std::string;

using dingofs::common::PartitionInfo;

class TopologyManager {
 public:
  TopologyManager(const std::shared_ptr<Topology>& topology,
                  std::shared_ptr<MetaserverClient> metaserver_client)
      : topology_(topology), metaserverClient_(metaserver_client) {}

  virtual ~TopologyManager() = default;

  virtual void Init(const TopologyOption& option);

  virtual void RegistMetaServer(const MetaServerRegistRequest* request,
                                MetaServerRegistResponse* response);

  virtual void ListMetaServer(const ListMetaServerRequest* request,
                              ListMetaServerResponse* response);

  virtual void GetMetaServer(const GetMetaServerInfoRequest* request,
                             GetMetaServerInfoResponse* response);

  virtual void DeleteMetaServer(const DeleteMetaServerRequest* request,
                                DeleteMetaServerResponse* response);

  virtual void RegistServer(const ServerRegistRequest* request,
                            ServerRegistResponse* response);

  virtual void GetServer(const GetServerRequest* request,
                         GetServerResponse* response);

  virtual void DeleteServer(const DeleteServerRequest* request,
                            DeleteServerResponse* response);

  virtual void ListZoneServer(const ListZoneServerRequest* request,
                              ListZoneServerResponse* response);

  virtual void CreateZone(const CreateZoneRequest* request,
                          CreateZoneResponse* response);

  virtual void DeleteZone(const DeleteZoneRequest* request,
                          DeleteZoneResponse* response);

  virtual void GetZone(const GetZoneRequest* request,
                       GetZoneResponse* response);

  virtual void ListPoolZone(const ListPoolZoneRequest* request,
                            ListPoolZoneResponse* response);

  virtual void CreatePool(const CreatePoolRequest* request,
                          CreatePoolResponse* response);

  virtual void DeletePool(const DeletePoolRequest* request,
                          DeletePoolResponse* response);

  virtual void GetPool(const GetPoolRequest* request,
                       GetPoolResponse* response);

  virtual void ListPool(const ListPoolRequest* request,
                        ListPoolResponse* response);

  virtual void CreatePartitions(const CreatePartitionRequest* request,
                                CreatePartitionResponse* response);

  virtual void DeletePartition(const DeletePartitionRequest* request,
                               DeletePartitionResponse* response);

  virtual TopoStatusCode CreatePartitionsAndGetMinPartition(
      FsIdType fs_id, PartitionInfo* partition);

  virtual TopoStatusCode DeletePartition(uint32_t partition_id);

  virtual TopoStatusCode CommitTxId(const std::vector<PartitionTxId>& tx_ids);

  virtual void CommitTx(const CommitTxRequest* request,
                        CommitTxResponse* response);

  virtual void GetMetaServerListInCopysets(
      const GetMetaServerListInCopySetsRequest* request,
      GetMetaServerListInCopySetsResponse* response);

  virtual void ListPartition(const ListPartitionRequest* request,
                             ListPartitionResponse* response);

  virtual void ListPartitionOfFs(FsIdType fs_id,
                                 std::list<PartitionInfo>* list);

  virtual void GetLatestPartitionsTxId(const std::vector<PartitionTxId>& tx_ids,
                                       std::vector<PartitionTxId>* need_update);

  virtual TopoStatusCode UpdatePartitionStatus(PartitionIdType partition_id,
                                               PartitionStatus status);

  virtual void GetCopysetOfPartition(
      const GetCopysetOfPartitionRequest* request,
      GetCopysetOfPartitionResponse* response);

  virtual TopoStatusCode GetCopysetMembers(PoolIdType pool_id,
                                           CopySetIdType copyset_id,
                                           std::set<std::string>* addrs);

  virtual bool CreateCopysetNodeOnMetaServer(PoolIdType pool_id,
                                             CopySetIdType copyset_id,
                                             MetaServerIdType meta_server_id);

  virtual void GetCopysetsInfo(const GetCopysetsInfoRequest* request,
                               GetCopysetsInfoResponse* response);

  virtual void ListCopysetsInfo(ListCopysetInfoResponse* response);

  virtual void GetMetaServersSpace(
      ::google::protobuf::RepeatedPtrField<
          dingofs::mds::topology::MetadataUsage>* spaces);

  virtual void GetTopology(ListTopologyResponse* response);

  virtual void ListZone(ListZoneResponse* response);

  virtual void ListServer(ListServerResponse* response);

  virtual void ListMetaserverOfCluster(ListMetaServerResponse* response);

  virtual void RegistMemcacheCluster(
      const RegistMemcacheClusterRequest* request,
      RegistMemcacheClusterResponse* response);

  virtual void ListMemcacheCluster(ListMemcacheClusterResponse* response);

  virtual void AllocOrGetMemcacheCluster(
      const AllocOrGetMemcacheClusterRequest* request,
      AllocOrGetMemcacheClusterResponse* response);

 private:
  TopoStatusCode CreateEnoughCopyset(int32_t create_num);

  TopoStatusCode CreateCopyset(const CopysetCreateInfo& copyset);

  virtual void GetCopysetInfo(const uint32_t& pool_id,
                              const uint32_t& copyset_id,
                              CopysetValue* copyset_value);

  virtual void ClearCopysetCreating(PoolIdType pool_id,
                                    CopySetIdType copyset_id);
  TopoStatusCode CreatePartitionOnCopyset(FsIdType fs_id,
                                          const CopySetInfo& copyset,
                                          PartitionInfo* info);

  std::shared_ptr<Topology> topology_;
  std::shared_ptr<MetaserverClient> metaserverClient_;

  /**
   * @brief register mutex for metaserver, preventing duplicate registration
   *        in concurrent scenario
   */
  NameLock registMsMutex_;

  /**
   * @brief register mutex for memcachecluster,
   *           preventing duplicate registration
   *        in concurrent scenario
   */
  mutable RWLock registMemcacheClusterMutex_;

  NameLock createPartitionMutex_;

  /**
   * @brief topology options
   */
  TopologyOption option_;
};

}  // namespace topology
}  // namespace mds
}  // namespace dingofs

#endif  // DINGOFS_SRC_MDS_TOPOLOGY_TOPOLOGY_MANAGER_H_