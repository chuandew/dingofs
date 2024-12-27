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
 * Created Date: Thur Jun 15 2021
 * Author: lixiaocui
 */

#include "dingofs/src/stub/rpcclient/mds_client.h"

#include <map>
#include <utility>
#include <vector>

#include "dingofs/proto/space.pb.h"
#include "dingofs/src/common/metric_utils.h"
#include "dingofs/src/stub/common/config.h"

namespace dingofs {
namespace stub {
namespace rpcclient {

using ::dingofs::mds::space::SpaceErrCode;
using ::dingofs::mds::space::SpaceErrCode_Name;
using ::dingofs::utils::TimeUtility;

using ::dingofs::stub::metric::MetricListGuard;

// rpc发送和mds地址切换状态机
int RPCExcutorRetryPolicy::DoRPCTask(RPCFunc rpctask, uint64_t maxRetryTimeMS) {
  // 记录上一次正在服务的mds index
  int lastWorkingMDSIndex = currentWorkingMDSAddrIndex_;

  // 记录当前正在使用的mds index
  int curRetryMDSIndex = currentWorkingMDSAddrIndex_;

  // 记录当前mds重试的次数
  uint64_t currentMDSRetryCount = 0;

  // 执行起始时间点
  uint64_t startTime = TimeUtility::GetTimeofDayMs();

  // rpc超时时间
  uint64_t rpcTimeOutMS = retryOpt_.rpcTimeoutMs;

  // The count of normal retry
  uint64_t normalRetryCount = 0;

  int retcode = -1;
  bool retryUnlimit = (maxRetryTimeMS == 0);
  while (GoOnRetry(startTime, maxRetryTimeMS)) {
    // 1. 创建当前rpc需要使用的channel和controller，执行rpc任务
    retcode = ExcuteTask(curRetryMDSIndex, rpcTimeOutMS, rpctask);

    // 2. 根据rpc返回值进行预处理
    if (retcode < 0) {
      curRetryMDSIndex = PreProcessBeforeRetry(
          retcode, retryUnlimit, &normalRetryCount, &currentMDSRetryCount,
          curRetryMDSIndex, &lastWorkingMDSIndex, &rpcTimeOutMS);
      continue;
      // 3. 此时rpc是正常返回的，更新当前正在服务的mds地址index
    } else {
      currentWorkingMDSAddrIndex_.store(curRetryMDSIndex);
      break;
    }
  }

  return retcode;
}

bool RPCExcutorRetryPolicy::GoOnRetry(uint64_t startTimeMS,
                                      uint64_t maxRetryTimeMS) {
  if (maxRetryTimeMS == 0) {
    return true;
  }

  uint64_t currentTime = TimeUtility::GetTimeofDayMs();
  return currentTime - startTimeMS < maxRetryTimeMS;
}

int RPCExcutorRetryPolicy::PreProcessBeforeRetry(int status, bool retryUnlimit,
                                                 uint64_t* normalRetryCount,
                                                 uint64_t* curMDSRetryCount,
                                                 int curRetryMDSIndex,
                                                 int* lastWorkingMDSIndex,
                                                 uint64_t* timeOutMS) {
  int nextMDSIndex = 0;
  bool rpcTimeout = false;
  bool needChangeMDS = false;

  // If retryUnlimit is set, sleep a long time to retry no matter what the
  // error it is.
  if (retryUnlimit) {
    if (++(*normalRetryCount) > retryOpt_.normalRetryTimesBeforeTriggerWait) {
      bthread_usleep(retryOpt_.waitSleepMs * 1000);
    }

    // 1. 访问存在的IP地址，但无人监听：ECONNREFUSED
    // 2. 正常发送RPC情况下，对端进程挂掉了：EHOSTDOWN
    // 3. 对端server调用了Stop：ELOGOFF
    // 4. 对端链接已关闭：ECONNRESET
    // 5. 在一个mds节点上rpc失败超过限定次数
    // 在这几种场景下，主动切换mds。
  } else if (status == -EHOSTDOWN || status == -ECONNRESET ||
             status == -ECONNREFUSED || status == -brpc::ELOGOFF ||
             *curMDSRetryCount >= retryOpt_.maxFailedTimesBeforeChangeAddr) {
    needChangeMDS = true;

    // 在开启健康检查的情况下，在底层tcp连接失败时
    // rpc请求会本地直接返回 EHOSTDOWN
    // 这种情况下，增加一些睡眠时间，避免大量的重试请求占满bthread
    // TODO(wuhanqing): 关闭健康检查
    if (status == -EHOSTDOWN) {
      bthread_usleep(retryOpt_.rpcRetryIntervalUS);
    }
  } else if (status == -brpc::ERPCTIMEDOUT || status == -ETIMEDOUT) {
    rpcTimeout = true;
    needChangeMDS = false;
    // 触发超时指数退避
    *timeOutMS *= 2;
    *timeOutMS = std::min(*timeOutMS, retryOpt_.maxRPCTimeoutMS);
    *timeOutMS = std::max(*timeOutMS, retryOpt_.rpcTimeoutMs);
  }

  // 获取下一次需要重试的mds索引
  nextMDSIndex = GetNextMDSIndex(needChangeMDS, curRetryMDSIndex,
                                 lastWorkingMDSIndex);  // NOLINT

  // 更新curMDSRetryCount和rpctimeout
  if (nextMDSIndex != curRetryMDSIndex) {
    *curMDSRetryCount = 0;
    *timeOutMS = retryOpt_.rpcTimeoutMs;
  } else {
    ++(*curMDSRetryCount);
    // 还是在当前mds上重试，且rpc不是超时错误，就进行睡眠，然后再重试
    if (!rpcTimeout) {
      bthread_usleep(retryOpt_.rpcRetryIntervalUS);
    }
  }

  return nextMDSIndex;
}
/**
 * 根据输入状态获取下一次需要重试的mds索引，mds切换逻辑：
 * 记录三个状态：curRetryMDSIndex、lastWorkingMDSIndex、
 *             currentWorkingMDSIndex
 * 1. 开始的时候curRetryMDSIndex = currentWorkingMDSIndex
 *            lastWorkingMDSIndex = currentWorkingMDSIndex
 * 2. 如果rpc失败，会触发切换curRetryMDSIndex，如果这时候lastWorkingMDSIndex
 *    与currentWorkingMDSIndex相等，这时候会顺序切换到下一个mds索引，
 *    如果lastWorkingMDSIndex与currentWorkingMDSIndex不相等，那么
 *    说明有其他接口更新了currentWorkingMDSAddrIndex_，那么本次切换
 *    直接切换到currentWorkingMDSAddrIndex_
 */
int RPCExcutorRetryPolicy::GetNextMDSIndex(bool needChangeMDS,
                                           int currentRetryIndex,
                                           int* lastWorkingindex) {
  int nextMDSIndex = 0;
  if (std::atomic_compare_exchange_strong(&currentWorkingMDSAddrIndex_,
                                          lastWorkingindex,
                                          currentWorkingMDSAddrIndex_.load())) {
    int size = retryOpt_.addrs.size();
    nextMDSIndex =
        needChangeMDS ? (currentRetryIndex + 1) % size : currentRetryIndex;
  } else {
    nextMDSIndex = *lastWorkingindex;
  }

  return nextMDSIndex;
}

int RPCExcutorRetryPolicy::ExcuteTask(int mdsindex, uint64_t rpcTimeOutMS,
                                      RPCFunc task) {
  const std::string& mdsaddr = retryOpt_.addrs[mdsindex];

  brpc::Channel channel;
  int ret = channel.Init(mdsaddr.c_str(), nullptr);
  if (ret != 0) {
    LOG(WARNING) << "Init channel failed! addr = " << mdsaddr;
    // 返回EHOSTDOWN给上层调用者，促使其切换mds
    return -EHOSTDOWN;
  }

  brpc::Controller cntl;
  cntl.set_log_id(GetLogId());
  cntl.set_timeout_ms(rpcTimeOutMS);

  return task(mdsindex, rpcTimeOutMS, &channel, &cntl);
}

FSStatusCode MdsClientImpl::Init(
    const ::dingofs::stub::common::MdsOption& mdsOpt,
    MDSBaseClient* baseclient) {
  mdsOpt_ = mdsOpt;
  rpcexcutor_.SetOption(mdsOpt_.rpcRetryOpt);
  mdsbasecli_ = baseclient;

  std::ostringstream oss;
  std::for_each(mdsOpt_.rpcRetryOpt.addrs.begin(),
                mdsOpt_.rpcRetryOpt.addrs.end(),
                [&](const std::string& addr) { oss << " " << addr; });

  LOG(INFO) << "MDSClient init success, addresses:" << oss.str();
  return FSStatusCode::OK;
}

#define RPCTask                                                     \
  [&](int addrindex, uint64_t rpctimeoutMS, brpc::Channel* channel, \
      brpc::Controller* cntl) -> int

FSStatusCode MdsClientImpl::MountFs(const std::string& fsName,
                                    const Mountpoint& mountPt, FsInfo* fsInfo) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // mount fs metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok, {&mdsClientMetric_.mountFs, &mdsClientMetric_.getAllOperation},
        start);

    MountFsResponse response;
    mdsbasecli_->MountFs(fsName, mountPt, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "MountFs Failed, errorcode = " << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    if (ret != FSStatusCode::OK) {
      LOG(WARNING) << "MountFs: fsname = " << fsName
                   << ", mountPt = " << mountPt.ShortDebugString()
                   << ", errcode = " << ret
                   << ", errmsg = " << FSStatusCode_Name(ret);
    } else if (response.has_fsinfo()) {
      fsInfo->CopyFrom(response.fsinfo());
    }
    return ret;
  };
  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::UmountFs(const std::string& fsName,
                                     const Mountpoint& mountPt) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // unmount fs metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok, {&mdsClientMetric_.umountFs, &mdsClientMetric_.getAllOperation},
        start);

    UmountFsResponse response;
    mdsbasecli_->UmountFs(fsName, mountPt, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "UmountFs Failed, errorcode = " << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    LOG_IF(WARNING, ret != FSStatusCode::OK)
        << "UmountFs: fsname = " << fsName
        << ", mountPt = " << mountPt.ShortDebugString() << ", errcode = " << ret
        << ", errmsg = " << FSStatusCode_Name(ret);
    return ret;
  };
  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::GetFsInfo(const std::string& fsName,
                                      FsInfo* fsInfo) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // getfsinfo metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.getFsInfo, &mdsClientMetric_.getAllOperation},
        start);

    GetFsInfoResponse response;
    mdsbasecli_->GetFsInfo(fsName, &response, cntl, channel);

    if (cntl->Failed()) {
      LOG(WARNING) << "GetFsInfo Failed, errorcode = " << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    if (ret != FSStatusCode::OK) {
      LOG(WARNING) << "GetFsInfo: fsname = " << fsName << ", errcode = " << ret
                   << ", errmsg = " << FSStatusCode_Name(ret);
    } else if (response.has_fsinfo()) {
      fsInfo->CopyFrom(response.fsinfo());
    }

    return ret;
  };
  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::GetFsInfo(uint32_t fsId, FsInfo* fsInfo) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // getfsinfo metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.getFsInfo, &mdsClientMetric_.getAllOperation},
        start);

    GetFsInfoResponse response;
    mdsbasecli_->GetFsInfo(fsId, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "GetFsInfo Failed, errorcode = " << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    if (ret != FSStatusCode::OK) {
      LOG(WARNING) << "GetFsInfo: fsid = " << fsId << ", errcode = " << ret
                   << ", errmsg = " << FSStatusCode_Name(ret);
    } else if (response.has_fsinfo()) {
      fsInfo->CopyFrom(response.fsinfo());
    }
    return ret;
  };
  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

template <typename T>
void GetEndPoint(const T& info, butil::EndPoint* internal,
                 butil::EndPoint* external) {
  const std::string& internalIp = info.internalip();
  const std::string& externalIp = [&info]() {
    if (info.has_externalip()) {
      return info.externalip();
    } else {
      return info.internalip();
    }
  }();

  const uint32_t internalPort = info.internalport();
  const uint32_t externalPort = [&info]() {
    if (info.has_externalport()) {
      return info.externalport();
    } else {
      return info.internalport();
    }
  }();

  butil::str2endpoint(internalIp.c_str(), internalPort, internal);
  butil::str2endpoint(externalIp.c_str(), externalPort, external);
}

bool MdsClientImpl::GetMetaServerInfo(
    const PeerAddr& addr, CopysetPeerInfo<MetaserverID>* metaserverInfo) {
  std::vector<std::string> strs;
  dingofs::utils::SplitString(addr.ToString(), ":", &strs);
  const std::string& ip = strs[0];
  uint64_t port;
  ::dingofs::utils::StringToUll(strs[1], &port);

  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    (void)addrindex;
    // getMetaServerInfo metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(&is_ok,
                             {&mdsClientMetric_.getMetaServerInfo,
                              &mdsClientMetric_.getAllOperation},
                             start);

    GetMetaServerInfoResponse response;
    mdsbasecli_->GetMetaServerInfo(port, ip, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "GetMetaServerInfo Failed, errorcode = "
                   << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    TopoStatusCode ret = response.statuscode();
    if (ret != TopoStatusCode::TOPO_OK) {
      LOG(WARNING) << "GetMetaServerInfo: ip= " << ip << ", port= " << port
                   << ", errcode = " << ret;
    } else {
      const auto& info = response.metaserverinfo();
      MetaserverID metaserverID = info.metaserverid();
      butil::EndPoint internal;
      butil::EndPoint external;
      GetEndPoint(info, &internal, &external);
      *metaserverInfo = CopysetPeerInfo<MetaserverID>(
          metaserverID, PeerAddr(internal), PeerAddr(external));
    }

    return ret;
  };
  return 0 == rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS);
}

bool MdsClientImpl::GetMetaServerListInCopysets(
    const LogicPoolID& logicalpooid, const std::vector<CopysetID>& copysetidvec,
    std::vector<CopysetInfo<MetaserverID>>* cpinfoVec) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // getMetaServerListInCopysets metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(&is_ok,
                             {&mdsClientMetric_.getMetaServerListInCopysets,
                              &mdsClientMetric_.getAllOperation},
                             start);

    GetMetaServerListInCopySetsResponse response;
    mdsbasecli_->GetMetaServerListInCopysets(logicalpooid, copysetidvec,
                                             &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "get metaserver list from mds failed, error is "
                   << cntl->ErrorText() << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    int csinfonum = response.csinfo_size();
    for (int i = 0; i < csinfonum; i++) {
      CopysetInfo<MetaserverID> copysetseverl;
      ::dingofs::mds::topology::CopySetServerInfo info = response.csinfo(i);

      copysetseverl.lpid_ = logicalpooid;
      copysetseverl.cpid_ = info.copysetid();
      int cslocsNum = info.cslocs_size();
      for (int j = 0; j < cslocsNum; j++) {
        CopysetPeerInfo<MetaserverID> csinfo;
        ::dingofs::mds::topology::MetaServerLocation csl = info.cslocs(j);
        csinfo.peerID = csl.metaserverid();
        butil::EndPoint internal;
        butil::EndPoint external;
        GetEndPoint(csl, &internal, &external);
        csinfo.internalAddr = PeerAddr(internal);
        csinfo.externalAddr = PeerAddr(external);
        copysetseverl.AddCopysetPeerInfo(csinfo);
      }
      cpinfoVec->push_back(copysetseverl);
    }
    TopoStatusCode ret = response.statuscode();
    LOG_IF(WARNING, TopoStatusCode::TOPO_OK != 0)
        << "GetMetaServerList failed"
        << ", errocde = " << response.statuscode()
        << ", log id = " << cntl->log_id();
    return ret;
  };

  return 0 == rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS);
}

bool MdsClientImpl::CreatePartition(
    uint32_t fsID, uint32_t count, std::vector<PartitionInfo>* partitionInfos) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // CreatePartition metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.createPartition, &mdsClientMetric_.getAllOperation},
        start);

    CreatePartitionResponse response;
    mdsbasecli_->CreatePartition(fsID, count, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "CreatePartition from mds failed, error is "
                   << cntl->ErrorText() << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    TopoStatusCode ret = response.statuscode();
    if (ret != TopoStatusCode::TOPO_OK) {
      LOG(WARNING) << "CreatePartition: fsID = " << fsID
                   << ", count = " << count << ", errcode = " << ret
                   << ", errmsg = " << TopoStatusCode_Name(ret);
      return ret;
    }

    int partitionNum = response.partitioninfolist_size();
    if (partitionNum == 0) {
      LOG(ERROR) << "CreatePartition: fsID = " << fsID << ", count = " << count
                 << ", errcode = " << ret
                 << ", errmsg = " << TopoStatusCode_Name(ret)
                 << ", but no partition info returns";
      return TopoStatusCode::TOPO_CREATE_PARTITION_FAIL;
    }

    partitionInfos->reserve(count);
    partitionInfos->clear();
    std::move(response.mutable_partitioninfolist()->begin(),
              response.mutable_partitioninfolist()->end(),
              std::back_inserter(*partitionInfos));

    return TopoStatusCode::TOPO_OK;
  };

  return 0 == rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS);
}

bool MdsClientImpl::GetCopysetOfPartitions(
    const std::vector<uint32_t>& partitionIDList,
    std::map<uint32_t, Copyset>* copysetMap) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // GetCopysetOfPartitions metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(&is_ok,
                             {&mdsClientMetric_.getCopysetOfPartitions,
                              &mdsClientMetric_.getAllOperation},
                             start);

    GetCopysetOfPartitionResponse response;
    mdsbasecli_->GetCopysetOfPartitions(partitionIDList, &response, cntl,
                                        channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "GetCopysetOfPartition from mds failed, error is "
                   << cntl->ErrorText() << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    TopoStatusCode ret = response.statuscode();
    if (ret != TopoStatusCode::TOPO_OK) {
      LOG(WARNING) << "GetCopysetOfPartition: errcode = " << ret
                   << ", errmsg = " << TopoStatusCode_Name(ret);
      return ret;
    }

    int size = response.copysetmap_size();
    if (size == 0) {
      LOG(WARNING) << "GetCopysetOfPartition: errcode = " << ret
                   << ", errmsg = " << TopoStatusCode_Name(ret)
                   << ", but no copyset returns";
      return TopoStatusCode::TOPO_INTERNAL_ERROR;
    }

    copysetMap->clear();
    for (auto it : response.copysetmap()) {
      CopysetPeerInfo<MetaserverID> csinfo;
      copysetMap->emplace(it.first, it.second);
    }

    return TopoStatusCode::TOPO_OK;
  };

  return 0 == rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS);
}

bool MdsClientImpl::ListPartition(uint32_t fsID,
                                  std::vector<PartitionInfo>* partitionInfos) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // listPartition metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.listPartition, &mdsClientMetric_.getAllOperation},
        start);

    ListPartitionResponse response;
    mdsbasecli_->ListPartition(fsID, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "ListPartition from mds failed, error is "
                   << cntl->ErrorText() << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    TopoStatusCode ret = response.statuscode();
    if (ret != TopoStatusCode::TOPO_OK) {
      LOG(WARNING) << "ListPartition: fsID = " << fsID << ", errcode = " << ret
                   << ", errmsg = " << TopoStatusCode_Name(ret);
      return ret;
    }

    partitionInfos->clear();
    // when fs is creating and mds exit at the same time,
    // this may cause this fs has no partition
    int partitionNum = response.partitioninfolist_size();
    for (int i = 0; i < partitionNum; i++) {
      partitionInfos->push_back(response.partitioninfolist(i));
    }

    return TopoStatusCode::TOPO_OK;
  };

  return 0 == rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS);
}

bool MdsClientImpl::AllocOrGetMemcacheCluster(uint32_t fsId,
                                              MemcacheClusterInfo* cluster) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // AllocOrGetMemcacheCluster metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(&is_ok,
                             {&mdsClientMetric_.allocOrGetMemcacheCluster,
                              &mdsClientMetric_.getAllOperation},
                             start);

    mds::topology::AllocOrGetMemcacheClusterResponse response;
    mdsbasecli_->AllocOrGetMemcacheCluster(fsId, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "AllocOrGetMemcacheCluster from mds failed, error is "
                   << cntl->ErrorText() << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    TopoStatusCode ret = response.statuscode();
    if (ret != TopoStatusCode::TOPO_OK) {
      LOG(WARNING) << "AllocOrGetMemcacheCluster fail, errcode = " << ret
                   << ", errmsg = " << TopoStatusCode_Name(ret);
      return ret;
    }

    *cluster = std::move(*response.mutable_cluster());

    return ret;
  };

  return 0 == ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::AllocS3ChunkId(uint32_t fsId, uint32_t idNum,
                                           uint64_t* chunkId) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // AllocS3ChunkId metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.allocS3ChunkId, &mdsClientMetric_.getAllOperation},
        start);

    AllocateS3ChunkResponse response;
    mdsbasecli_->AllocS3ChunkId(fsId, idNum, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "AllocS3ChunkId Failed, errorcode = " << cntl->ErrorCode()
                   << ", error content:" << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    if (ret != FSStatusCode::OK) {
      LOG(WARNING) << "AllocS3ChunkId: fsid = " << fsId << ", errcode = " << ret
                   << ", errmsg = " << FSStatusCode_Name(ret);
    } else if (response.has_beginchunkid()) {
      *chunkId = response.beginchunkid();
    }

    return ret;
  };
  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::RefreshSession(
    const std::vector<PartitionTxId>& txIds,
    std::vector<PartitionTxId>* latestTxIdList, const std::string& fsName,
    const Mountpoint& mountpoint, std::atomic<bool>* enableSumInDir) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    // RefreshSession metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.refreshSession, &mdsClientMetric_.getAllOperation},
        start);

    RefreshSessionRequest request;
    RefreshSessionResponse response;
    *request.mutable_txids() = {txIds.begin(), txIds.end()};
    request.set_fsname(fsName);
    *request.mutable_mountpoint() = mountpoint;
    mdsbasecli_->RefreshSession(request, &response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "RefreshSession fail, errcode = " << cntl->ErrorCode()
                   << ", error content: " << cntl->ErrorText()
                   << ", log id = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode ret = response.statuscode();
    if (ret != FSStatusCode::OK) {
      LOG(WARNING) << "RefreshSession fail, errcode = " << ret
                   << ", errmsg = " << FSStatusCode_Name(ret);
    } else if (response.latesttxidlist_size() > 0) {
      *latestTxIdList = {response.latesttxidlist().begin(),
                         response.latesttxidlist().end()};
      LOG(INFO) << "RefreshSession need update partition txid list: "
                << response.DebugString();
    }
    if (enableSumInDir->load() && !response.enablesumindir()) {
      enableSumInDir->store(response.enablesumindir());
      LOG(INFO) << "update enableSumInDir to " << response.enablesumindir();
    }

    return ret;
  };

  return ReturnError(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

FSStatusCode MdsClientImpl::GetLatestTxId(const GetLatestTxIdRequest& request,
                                          GetLatestTxIdResponse* response) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    VLOG(12) << "GetLatestTxId [request]: " << request.ShortDebugString();
    // GetLatestTxId metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok,
        {&mdsClientMetric_.getLatestTxId, &mdsClientMetric_.getAllOperation},
        start);

    mdsbasecli_->GetLatestTxId(request, response, cntl, channel);
    if (cntl->Failed()) {
      LOG(WARNING) << "GetLatestTxId fail, errCode = " << cntl->ErrorCode()
                   << ", errorText = " << cntl->ErrorText()
                   << ", logId = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode rc = response->statuscode();
    if (rc == FSStatusCode::LOCK_FAILED) {
      LOG(WARNING) << "GetLatestTxId fail for acquire dlock failed";
      return -rc;
    } else if (rc == FSStatusCode::LOCK_TIMEOUT) {
      LOG(WARNING) << "GetLatestTxId fail for acquire dlock timeout";
      return -rc;
    } else if (rc != FSStatusCode::OK) {
      LOG(WARNING) << "GetLatestTxId fail, errcode = " << rc
                   << ", errmsg = " << FSStatusCode_Name(rc);
    }

    VLOG(12) << "GetLatestTxId [response]: " << response->ShortDebugString();
    return rc;
  };

  // for rpc error or get lock failed/timeout, we will retry until success
  return ReturnError(rpcexcutor_.DoRPCTask(task, 0));
}

FSStatusCode MdsClientImpl::CommitTx(const CommitTxRequest& request) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    VLOG(12) << "CommitTx [request]: " << request.DebugString();
    // CommitTx metrics information
    auto start = butil::cpuwide_time_us();
    bool is_ok = true;
    MetricListGuard mdsGuard(
        &is_ok, {&mdsClientMetric_.commitTx, &mdsClientMetric_.getAllOperation},
        start);

    CommitTxResponse response;
    mdsbasecli_->CommitTx(request, &response, cntl, channel);

    if (cntl->Failed()) {
      LOG(WARNING) << "CommitTx failed, errorCode = " << cntl->ErrorCode()
                   << ", errorText =" << cntl->ErrorText()
                   << ", logId = " << cntl->log_id();
      is_ok = false;
      return -cntl->ErrorCode();
    }

    FSStatusCode rc = response.statuscode();
    if (rc == FSStatusCode::LOCK_FAILED) {
      LOG(WARNING) << "CommitTx fail for acquire dlock failed";
      return -rc;
    } else if (rc == FSStatusCode::LOCK_TIMEOUT) {
      LOG(WARNING) << "CommitTx fail for acquire dlock timeout";
      return -rc;
    } else if (rc != FSStatusCode::OK) {
      LOG(WARNING) << "CommitTx: retCode = " << rc
                   << ", message = " << FSStatusCode_Name(rc);
    }
    VLOG(12) << "CommitTx [response]: " << response.DebugString();
    return rc;
  };
  // for rpc error or get lock failed/timeout, we will retry until success
  return ReturnError(rpcexcutor_.DoRPCTask(task, 0));
}

FSStatusCode MdsClientImpl::GetLatestTxId(uint32_t fsId,
                                          std::vector<PartitionTxId>* txIds) {
  GetLatestTxIdRequest request;
  GetLatestTxIdResponse response;
  request.set_fsid(fsId);
  FSStatusCode rc = GetLatestTxId(request, &response);
  if (rc == FSStatusCode::OK) {
    *txIds = {response.txids().begin(), response.txids().end()};
  }
  return rc;
}

FSStatusCode MdsClientImpl::GetLatestTxIdWithLock(
    uint32_t fsId, const std::string& fsName, const std::string& uuid,
    std::vector<PartitionTxId>* txIds, uint64_t* txSequence) {
  GetLatestTxIdRequest request;
  GetLatestTxIdResponse response;
  request.set_lock(true);
  request.set_fsid(fsId);
  request.set_fsname(fsName);
  request.set_uuid(uuid);
  FSStatusCode rc = GetLatestTxId(request, &response);
  if (rc == FSStatusCode::OK) {
    *txIds = {response.txids().begin(), response.txids().end()};
    *txSequence = response.txsequence();
  }
  return rc;
}

FSStatusCode MdsClientImpl::CommitTx(const std::vector<PartitionTxId>& txIds) {
  CommitTxRequest request;
  *request.mutable_partitiontxids() = {txIds.begin(), txIds.end()};
  return CommitTx(request);
}

FSStatusCode MdsClientImpl::CommitTxWithLock(
    const std::vector<PartitionTxId>& txIds, const std::string& fsName,
    const std::string& uuid, uint64_t sequence) {
  CommitTxRequest request;
  request.set_lock(true);
  request.set_fsname(fsName);
  request.set_uuid(uuid);
  request.set_txsequence(sequence);
  *request.mutable_partitiontxids() = {txIds.begin(), txIds.end()};
  return CommitTx(request);
}

FSStatusCode MdsClientImpl::ReturnError(int retcode) {
  // rpc error convert to FSStatusCode::RPC_ERROR
  if (retcode < 0) {
    return FSStatusCode::RPC_ERROR;
  }

  // logic error
  return static_cast<FSStatusCode>(retcode);
}

static SpaceErrCode ToSpaceErrCode(int err) {
  if (err < 0) {
    return SpaceErrCode::SpaceErrUnknown;
  }

  return static_cast<SpaceErrCode>(err);
}

#define CHECK_RPC_AND_RETRY_IF_ERROR(msg)                            \
  do {                                                               \
    if (cntl->Failed()) {                                            \
      LOG(WARNING) << msg << " failed, error: " << cntl->ErrorText() \
                   << ", log id: " << cntl->log_id();                \
      return -cntl->ErrorCode();                                     \
    }                                                                \
  } while (0)

SpaceErrCode MdsClientImpl::AllocateVolumeBlockGroup(
    uint32_t fsId, uint32_t count, const std::string& owner,
    std::vector<dingofs::mds::space::BlockGroup>* groups) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    AllocateBlockGroupResponse response;
    mdsbasecli_->AllocateVolumeBlockGroup(fsId, count, owner, &response, cntl,
                                          channel);

    CHECK_RPC_AND_RETRY_IF_ERROR("AllocateVolumeBlockGroup");

    auto status = response.status();
    if (status != SpaceErrCode::SpaceOk) {
      LOG(WARNING) << "Allocate volume block group failed, err: "
                   << SpaceErrCode_Name(status);
    } else if (response.blockgroups_size() == 0) {
      LOG(WARNING) << "Allocate volume block group failed, no block "
                      "group allcoated";
      return SpaceErrCode::SpaceErrNoSpace;
    } else {
      VLOG(9) << "AllocateVolumeBlockGroup, response: "
              << response.ShortDebugString();
      groups->reserve(response.blockgroups_size());
      std::move(response.mutable_blockgroups()->begin(),
                response.mutable_blockgroups()->end(),
                std::back_inserter(*groups));
    }

    return status;
  };

  return ToSpaceErrCode(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

SpaceErrCode MdsClientImpl::AcquireVolumeBlockGroup(
    uint32_t fsId, uint64_t blockGroupOffset, const std::string& owner,
    dingofs::mds::space::BlockGroup* groups) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    AcquireBlockGroupResponse response;
    mdsbasecli_->AcquireVolumeBlockGroup(fsId, blockGroupOffset, owner,
                                         &response, cntl, channel);

    CHECK_RPC_AND_RETRY_IF_ERROR("AcquireVolumeBlockGroup");

    auto status = response.status();
    if (status != SpaceErrCode::SpaceOk) {
      LOG(WARNING) << "Acquire volume block group failed, err: "
                   << SpaceErrCode_Name(status);
    } else {
      groups->Swap(response.mutable_blockgroups());
    }

    return status;
  };

  return ToSpaceErrCode(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

SpaceErrCode MdsClientImpl::ReleaseVolumeBlockGroup(
    uint32_t fsId, const std::string& owner,
    const std::vector<dingofs::mds::space::BlockGroup>& blockGroups) {
  auto task = RPCTask {
    (void)addrindex;
    (void)rpctimeoutMS;
    ReleaseBlockGroupResponse response;
    mdsbasecli_->ReleaseVolumeBlockGroup(fsId, owner, blockGroups, &response,
                                         cntl, channel);

    CHECK_RPC_AND_RETRY_IF_ERROR("ReleaseVolumeBlockGroup");

    LOG_IF(WARNING, SpaceErrCode::SpaceOk != response.status())
        << "Release volume block group failed, err: "
        << SpaceErrCode_Name(response.status());

    return response.status();
  };

  return ToSpaceErrCode(rpcexcutor_.DoRPCTask(task, mdsOpt_.mdsMaxRetryMS));
}

#undef CHECK_RPC_AND_RETRY_IF_ERROR

}  // namespace rpcclient
}  // namespace stub
}  // namespace dingofs