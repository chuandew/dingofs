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
 * Created Date: 2021-05-19
 * Author: chenwei
 */

#ifndef DINGOFS_SRC_METASERVER_DENTRY_MANAGER_H_
#define DINGOFS_SRC_METASERVER_DENTRY_MANAGER_H_

#include <atomic>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "dingofs/proto/metaserver.pb.h"
#include "dingofs/src/metaserver/dentry_storage.h"
#include "dingofs/src/metaserver/transaction.h"

namespace dingofs {
namespace metaserver {
class DentryManager {
 public:
  DentryManager(std::shared_ptr<DentryStorage> dentryStorage,
                std::shared_ptr<TxManager> txManger);

  MetaStatusCode CreateDentry(const Dentry& dentry);

  // only invoked from snapshot loadding
  MetaStatusCode CreateDentry(const DentryVec& vec, bool merge);

  MetaStatusCode DeleteDentry(const Dentry& dentry);

  MetaStatusCode GetDentry(Dentry* dentry);

  MetaStatusCode ListDentry(const Dentry& dentry, std::vector<Dentry>* dentrys,
                            uint32_t limit, bool onlyDir = false);

  void ClearDentry();

  MetaStatusCode HandleRenameTx(const std::vector<Dentry>& dentrys);

 private:
  void Log4Dentry(const std::string& request, const Dentry& dentry);
  void Log4Code(const std::string& request, MetaStatusCode rc);

 private:
  std::shared_ptr<DentryStorage> dentryStorage_;
  std::shared_ptr<TxManager> txManager_;
};

}  // namespace metaserver
}  // namespace dingofs

#endif  // DINGOFS_SRC_METASERVER_DENTRY_MANAGER_H_