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
 * File Created: Saturday, 13th October 2018 1:59:08 pm
 * Author: tongguangxun
 */
#include <fcntl.h>  // NOLINT
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <chrono>  // NOLINT
#include <iostream>
#include <string>
#include <thread>  // NOLINT

#include "include/client/libcurve.h"
#include "src/client/client_common.h"
#include "src/client/file_instance.h"
#include "test/client/fake/fakeMDS.h"
#include "test/client/fake/mock_schedule.h"

using curve::client::EndPoint;
using curve::client::PeerAddr;

uint32_t segment_size = 1 * 1024 * 1024 * 1024ul;  // NOLINT
uint32_t chunk_size = 16 * 1024 * 1024;            // NOLINT
std::string mdsMetaServerAddr = "127.0.0.1:9104";  // NOLINT

DECLARE_uint64(test_disk_size);
DEFINE_uint32(io_time, 5, "Duration for I/O test");
DEFINE_bool(fake_mds, true, "create fake mds");
DEFINE_bool(create_copysets, true, "create copysets on chunkserver");
DEFINE_bool(verify_io, true, "verify read/write I/O getting done correctly");

bool writeflag = false;
bool readflag = false;
std::mutex writeinterfacemtx;
std::condition_variable writeinterfacecv;
std::mutex interfacemtx;
std::condition_variable interfacecv;

DECLARE_uint64(test_disk_size);
void writecallbacktest(CurveAioContext* context) {
  writeflag = true;
  writeinterfacecv.notify_one();
  delete context;
}
void readcallbacktest(CurveAioContext* context) {
  readflag = true;
  interfacecv.notify_one();
  delete context;
}

int main(int argc, char** argv) {
  // google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  std::string configpath = "./test/client/configs/client.conf";

  if (Init(configpath.c_str()) != 0) {
    LOG(FATAL) << "Fail to init config";
  }

  // filename必须是全路径
  std::string filename = "/1_userinfo_";
  // uint64_t size = FLAGS_test_disk_size;

  /*** init mds service ***/
  FakeMDS mds(filename);
  if (FLAGS_fake_mds) {
    mds.Initialize();
    mds.StartService();
    if (FLAGS_create_copysets) {
      // 设置leaderid
      EndPoint ep;
      butil::str2endpoint("127.0.0.1", 9106, &ep);
      PeerId pd(ep);
      mds.StartCliService(pd);
      mds.CreateCopysetNode(true);
    }
  }

  /**** libcurve file operation ****/
  C_UserInfo_t userinfo;
  memcpy(userinfo.owner, "userinfo", 9);
  Create(filename.c_str(), &userinfo, FLAGS_test_disk_size);

  sleep(1);

  int fd;
  char* buffer;
  char* readbuffer;
  if (!FLAGS_verify_io) {
    // goto skip_write_io;
  }

  fd = Open(filename.c_str(), &userinfo);

  if (fd == -1) {
    LOG(FATAL) << "open file failed!";
    return -1;
  }

  buffer = new char[8 * 1024];
  memset(buffer, 'a', 1024);
  memset(buffer + 1024, 'b', 1024);
  memset(buffer + 2 * 1024, 'c', 1024);
  memset(buffer + 3 * 1024, 'd', 1024);
  memset(buffer + 4 * 1024, 'e', 1024);
  memset(buffer + 5 * 1024, 'f', 1024);
  memset(buffer + 6 * 1024, 'g', 1024);
  memset(buffer + 7 * 1024, 'h', 1024);

  uint64_t offset_base;
  for (int i = 0; i < 16; i++) {
    uint64_t offset = i * chunk_size;
    Write(fd, buffer, offset, 4096);
  }

  char* buf2 = new char[128 * 1024];
  char* buf1 = new char[128 * 1024];

  auto f = [&]() {
    uint64_t j = 0;
    // mds.EnableNetUnstable(600);
    while (1) {
      for (int i = 0; i < 10; i++) {
        CurveAioContext* aioctx2 = new CurveAioContext;
        CurveAioContext* aioctx1 = new CurveAioContext;
        aioctx1->buf = buf1;
        aioctx1->offset = 0;
        aioctx1->op = LIBCURVE_OP_WRITE;
        aioctx1->length = 128 * 1024;
        aioctx1->cb = writecallbacktest;
        AioWrite(fd, aioctx1);
        aioctx2->buf = buf2;
        aioctx2->offset = 0;
        aioctx2->length = 128 * 1024;
        aioctx2->op = LIBCURVE_OP_READ;
        aioctx2->cb = readcallbacktest;
        AioRead(fd, aioctx2);
        if (j % 10 == 0) {
          mds.EnableNetUnstable(600);
        } else {
          mds.EnableNetUnstable(100);
        }
        j++;
      }

      // char buf1[4096];
      // char buf2[8192];
      // Write(fd, buf2, 0, 8192);
      // Read(fd, buf1, 0, 4096);
    }
  };

  std::thread t1(f);
  std::thread t2(f);

  t1.join();
  t2.join();

  delete[] buf1;
  delete[] buf2;
  // delete aioctx2;
  // delete aioctx1;

  CurveAioContext writeaioctx;
  CurveAioContext readaioctx;
  {
    std::unique_lock<std::mutex> lk(writeinterfacemtx);
    writeinterfacecv.wait(lk, []() -> bool { return writeflag; });
  }
  writeflag = false;
  AioWrite(fd, &writeaioctx);
  {
    std::unique_lock<std::mutex> lk(writeinterfacemtx);
    writeinterfacecv.wait(lk, []() -> bool { return writeflag; });
  }

  {
    std::unique_lock<std::mutex> lk(interfacemtx);
    interfacecv.wait(lk, []() -> bool { return readflag; });
  }

  for (int i = 0; i < 1024; i++) {
    if (readbuffer[i] != 'a') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 1024] != 'b') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 2 * 1024] != 'c') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 3 * 1024] != 'd') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 4 * 1024] != 'e') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 5 * 1024] != 'f') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 6 * 1024] != 'g') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
    if (readbuffer[i + 7 * 1024] != 'h') {
      LOG(FATAL) << "read wrong data!";
      break;
    }
  }

  LOG(INFO) << "LibCurve I/O verified for stage 1, going to read repeatedly";

  // skip_write_io:
  std::atomic<bool> stop(false);
  auto testfunc = [&]() {
    while (!stop.load()) {
      if (!FLAGS_verify_io) {
        goto skip_read_io;
      }
      readflag = false;
      AioRead(fd, &readaioctx);
      {
        std::unique_lock<std::mutex> lk(interfacemtx);
        interfacecv.wait(lk, []() -> bool { return readflag; });
      }
      for (int i = 0; i < 1024; i++) {
        if (readbuffer[i] != 'a') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 1024] != 'b') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 2 * 1024] != 'c') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 3 * 1024] != 'd') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 4 * 1024] != 'e') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 5 * 1024] != 'f') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 6 * 1024] != 'g') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
        if (readbuffer[i + 7 * 1024] != 'h') {
          LOG(FATAL) << "read wrong data!";
          break;
        }
      }

    skip_read_io:
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  };

  std::thread t(testfunc);
  char c;

  if (FLAGS_io_time > 0) {
    sleep(FLAGS_io_time);
  } else {
    std::cin >> c;
  }

  stop.store(true);
  if (t.joinable()) {
    t.join();
  }

  Close(fd);
  UnInit();

  if (FLAGS_fake_mds) {
    mds.UnInitialize();
  }

  if (!FLAGS_verify_io) {
    goto workflow_finish;
  }
  delete[] buffer;
  delete[] readbuffer;

workflow_finish:
  unlink(configpath.c_str());
  return 0;
}
