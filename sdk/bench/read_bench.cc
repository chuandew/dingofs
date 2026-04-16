/*
 * Copyright (c) 2026 dingodb.com, Inc. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * DingoFS Read Benchmark — bypass FUSE, call libdingofs C API directly.
 *
 * This tool stress-tests the client-side read path (VFS -> FileReader ->
 * ChunkReader -> BlockStore) without FUSE kernel overhead.
 *
 * Workflow:
 *   1. Pre-fill: each thread writes its own data file (independent dir)
 *   2. (Optional) Sleep to let writeback / cache settle
 *   3. Read phase: each thread sequentially or randomly reads its file
 *   4. Cleanup: remove files (optional)
 *
 * Usage:
 *   # Sequential read
 *   read_bench --bench_mds_addr=10.0.0.1:8801 --bench_fs_name=myfs \
 *              --bench_threads=8 --bench_file_size_mb=1024
 *
 *   # Random read with FakeAccesser (isolate cache layer)
 *   read_bench --bench_mds_addr=10.0.0.1:8801 --bench_fs_name=myfs \
 *              --bench_threads=16 --bench_file_size_mb=2048 \
 *              --bench_pattern=random --bench_fake_access
 *
 *   # Skip prefill (data already exists from previous run)
 *   read_bench [...] --bench_skip_prefill
 */

#include <fcntl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "c/libdingofs.h"

// ---------------------------------------------------------------------------
// gflags
// ---------------------------------------------------------------------------

DEFINE_string(bench_mds_addr, "",
              "MDS address (e.g. 10.0.0.1:8801) or local://fs");
DEFINE_string(bench_fs_name, "", "Filesystem name");
DEFINE_string(bench_mount_point, "/bench_mount", "Logical mount point label");
DEFINE_string(bench_conf, "", "Config file path (gflags format)");
DEFINE_string(bench_dir, "/read_bench", "Directory inside FS for bench files");

DEFINE_int32(bench_threads, 1, "Number of concurrent reader threads");
DEFINE_int32(bench_file_size_mb, 1024, "File size per thread in MB");
DEFINE_int32(bench_block_size_kb, 4096, "Read block size in KB (default 4MB)");

DEFINE_string(bench_pattern, "sequential", "Read pattern: sequential|random");
DEFINE_bool(bench_skip_prefill, false,
            "Skip prefill phase (data must already exist)");
DEFINE_int32(bench_prefill_settle_sec, 0,
             "Sleep N seconds after prefill before reading (let cache settle)");

DEFINE_bool(bench_fake_blockstore, false,
            "Use FakeBlockStore to bypass BlockStore+Cache+IO entirely");
DEFINE_bool(bench_fake_access, false,
            "Use FakeAccesser to bypass IO only, keep TierBlockCache path");
DEFINE_bool(bench_cleanup, true, "Remove bench files after completion");
DEFINE_bool(bench_progress, true, "Show live progress during benchmark");

DEFINE_string(bench_log_dir, "/tmp", "Log directory for glog");
DEFINE_string(bench_log_level, "WARNING",
              "Log level: INFO/WARNING/ERROR/FATAL");

DEFINE_int32(bench_read_buffer_mb, 0,
             "Override vfs_read_buffer_total_mb (0 = use default)");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double NowSec() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static std::string HumanSize(size_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int idx = 0;
  double val = static_cast<double>(bytes);
  while (val >= 1024.0 && idx < 4) {
    val /= 1024.0;
    idx++;
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "%.1f %s", val, units[idx]);
  return buf;
}

// ---------------------------------------------------------------------------
// Per-thread result
// ---------------------------------------------------------------------------

struct ThreadResult {
  int thread_id;
  size_t bytes_read;
  double elapsed_sec;
  double throughput_mbs;
  int error_code;
};

// ---------------------------------------------------------------------------
// Progress tracker
// ---------------------------------------------------------------------------

class ProgressTracker {
 public:
  ProgressTracker(size_t total_bytes, int num_threads, const char* phase)
      : total_bytes_(total_bytes),
        num_threads_(num_threads),
        phase_(phase),
        progressed_(0),
        running_(true) {
    if (FLAGS_bench_progress) {
      thread_ = std::thread([this]() { Run(); });
    }
  }

  ~ProgressTracker() { Stop(); }

  void Add(size_t bytes) {
    progressed_.fetch_add(bytes, std::memory_order_relaxed);
  }

  void Stop() {
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  void Run() {
    auto start = std::chrono::steady_clock::now();
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (!running_) break;

      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      size_t p = progressed_.load(std::memory_order_relaxed);
      double pct = 100.0 * p / total_bytes_;
      double mbs = (p / (1024.0 * 1024.0)) / elapsed;

      std::cout << "\r  [" << phase_ << " " << num_threads_ << "T] "
                << std::fixed << std::setprecision(1) << pct << "%  "
                << HumanSize(p) << " / " << HumanSize(total_bytes_) << "  "
                << std::setprecision(1) << mbs << " MB/s  " << std::flush;
    }
  }

  size_t total_bytes_;
  int num_threads_;
  const char* phase_;
  std::atomic<size_t> progressed_;
  std::atomic<bool> running_;
  std::thread thread_;
};

// ---------------------------------------------------------------------------
// Prefill thread (write data so we have something to read)
// ---------------------------------------------------------------------------

static void PrefillThread(uintptr_t handle, int thread_id, size_t file_size,
                          size_t block_size, ProgressTracker* progress,
                          int* error_code) {
  *error_code = 0;

  char path[512];
  snprintf(path, sizeof(path), "%s/%d/data", FLAGS_bench_dir.c_str(),
           thread_id);

  int fd = dingofs_open(handle, path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "prefill thread " << thread_id << ": open failed: " << fd
              << std::endl;
    *error_code = fd;
    return;
  }

  std::vector<char> buf(block_size);
  // Write a recognizable pattern (not all zeros — avoid sparse-file traps)
  std::memset(buf.data(), 'A' + (thread_id % 26), block_size);

  size_t written = 0;
  while (written < file_size) {
    size_t to_write = std::min(block_size, file_size - written);
    ssize_t n = dingofs_pwrite(handle, fd, buf.data(), to_write, written);
    if (n < 0) {
      *error_code = static_cast<int>(n);
      break;
    }
    written += n;
    if (progress) progress->Add(n);
  }

  dingofs_flush(handle, fd);
  dingofs_close(handle, fd);
}

// ---------------------------------------------------------------------------
// Reader thread
// ---------------------------------------------------------------------------

static void ReaderThread(uintptr_t handle, int thread_id, size_t file_size,
                         size_t block_size, bool random_pattern,
                         ProgressTracker* progress, ThreadResult* result) {
  result->thread_id = thread_id;
  result->bytes_read = 0;
  result->elapsed_sec = 0;
  result->throughput_mbs = 0;
  result->error_code = 0;

  char path[512];
  snprintf(path, sizeof(path), "%s/%d/data", FLAGS_bench_dir.c_str(),
           thread_id);

  // O_RDONLY is 0 which MDS rejects as "flags is empty"; add O_NOATIME
  // (0x40000)
  int fd = dingofs_open(handle, path, O_RDONLY | O_NOATIME, 0);
  if (fd < 0) {
    std::cerr << "thread " << thread_id << ": open failed: " << fd << std::endl;
    result->error_code = fd;
    return;
  }

  std::vector<char> buf(block_size);

  // Build read offset list
  size_t num_blocks = (file_size + block_size - 1) / block_size;
  std::vector<size_t> offsets;
  offsets.reserve(num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    offsets.push_back(i * block_size);
  }
  if (random_pattern) {
    std::mt19937 rng(0xC0FFEE + thread_id);
    std::shuffle(offsets.begin(), offsets.end(), rng);
  }

  double t0 = NowSec();
  size_t total_read = 0;
  for (size_t off : offsets) {
    size_t to_read = std::min(block_size, file_size - off);
    ssize_t n = dingofs_pread(handle, fd, buf.data(), to_read, off);
    if (n < 0) {
      std::cerr << "thread " << thread_id << ": read failed at offset " << off
                << ": " << n << std::endl;
      result->error_code = static_cast<int>(n);
      break;
    }
    if (n == 0) break;  // EOF
    total_read += n;
    if (progress) progress->Add(n);
  }

  dingofs_close(handle, fd);
  double t1 = NowSec();

  result->bytes_read = total_read;
  result->elapsed_sec = t1 - t0;
  result->throughput_mbs =
      (total_read / (1024.0 * 1024.0)) / result->elapsed_sec;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void PrintUsage(const char* prog) {
  std::cout << R"(
DingoFS Read Benchmark — bypass FUSE, call libdingofs C API directly.

Usage:
  )" << prog << R"( --bench_mds_addr=ADDR --bench_fs_name=NAME [options]

Required:
  --bench_mds_addr        MDS address or local:// URL
  --bench_fs_name         Filesystem name

Options:
  --bench_threads=N       Number of reader threads (default: 1)
  --bench_file_size_mb=N  File size per thread in MB (default: 1024)
  --bench_block_size_kb=N Read block size in KB (default: 4096 = 4MB)
  --bench_pattern=P       sequential | random  (default: sequential)
  --bench_skip_prefill    Skip prefill phase (data must already exist)
  --bench_prefill_settle_sec=N  Sleep after prefill (default: 0)
  --bench_dir=PATH        FS directory for bench files (default: /read_bench)
  --bench_fake_blockstore Bypass BlockStore + Cache + IO entirely
  --bench_fake_access     Bypass IO only, keep TierBlockCache path
  --bench_cleanup         Remove files after test (default: true)

Examples:
  )" << prog << R"( --bench_mds_addr=10.0.0.1:8801 --bench_fs_name=myfs \
      --bench_threads=8 --bench_file_size_mb=1024 --bench_pattern=sequential

  )" << prog << R"( --bench_mds_addr=10.0.0.1:8801 --bench_fs_name=myfs \
      --bench_threads=16 --bench_file_size_mb=2048 --bench_fake_access
)";
}

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("DingoFS Read Benchmark (bypass FUSE)");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_bench_mds_addr.empty() || FLAGS_bench_fs_name.empty()) {
    PrintUsage(argv[0]);
    return 1;
  }

  bool random_pattern = (FLAGS_bench_pattern == "random");
  if (FLAGS_bench_pattern != "sequential" && FLAGS_bench_pattern != "random") {
    std::cerr << "Invalid --bench_pattern: " << FLAGS_bench_pattern
              << " (must be sequential|random)" << std::endl;
    return 1;
  }

  size_t file_size =
      static_cast<size_t>(FLAGS_bench_file_size_mb) * 1024 * 1024;
  size_t block_size = static_cast<size_t>(FLAGS_bench_block_size_kb) * 1024;
  int nthreads = FLAGS_bench_threads;

  std::cout << "DingoFS Read Benchmark (bypass FUSE)" << std::endl;
  std::cout << "  mds_addr:       " << FLAGS_bench_mds_addr << std::endl;
  std::cout << "  fs_name:        " << FLAGS_bench_fs_name << std::endl;
  std::cout << "  threads:        " << nthreads << std::endl;
  std::cout << "  file_size:      " << FLAGS_bench_file_size_mb << " MB"
            << std::endl;
  std::cout << "  block_size:     " << FLAGS_bench_block_size_kb << " KB"
            << std::endl;
  std::cout << "  pattern:        " << FLAGS_bench_pattern << std::endl;
  std::cout << "  total_read:     "
            << HumanSize(static_cast<size_t>(nthreads) * file_size)
            << std::endl;
  std::cout << "  fake_blockstore:"
            << (FLAGS_bench_fake_blockstore ? " yes" : " no") << std::endl;
  std::cout << "  fake_access:    " << (FLAGS_bench_fake_access ? "yes" : "no")
            << std::endl;
  std::cout << "  skip_prefill:   " << (FLAGS_bench_skip_prefill ? "yes" : "no")
            << std::endl;
  if (FLAGS_bench_prefill_settle_sec > 0) {
    std::cout << "  settle:         " << FLAGS_bench_prefill_settle_sec << " s"
              << std::endl;
  }
  std::cout << std::endl;

  uintptr_t h = dingofs_new();
  dingofs_conf_set(h, "log.dir", FLAGS_bench_log_dir.c_str());
  dingofs_conf_set(h, "log.level", FLAGS_bench_log_level.c_str());

  if (!FLAGS_bench_conf.empty()) {
    int rc = dingofs_conf_load(h, FLAGS_bench_conf.c_str());
    if (rc != 0) {
      std::cerr << "Failed to load config " << FLAGS_bench_conf << ": " << rc
                << std::endl;
      dingofs_delete(h);
      return 1;
    }
  }

  if (FLAGS_bench_fake_blockstore) {
    dingofs_conf_set(h, "vfs_use_fake_block_store", "true");
    std::cout << "FakeBlockStore: enabled" << std::endl;
  }
  if (FLAGS_bench_fake_access) {
    dingofs_conf_set(h, "use_fake_block_access", "true");
    std::cout << "FakeAccesser: enabled" << std::endl;
  }
  if (FLAGS_bench_read_buffer_mb > 0) {
    dingofs_conf_set(h, "vfs_read_buffer_total_mb",
                     std::to_string(FLAGS_bench_read_buffer_mb).c_str());
    std::cout << "Read buffer total: " << FLAGS_bench_read_buffer_mb << " MB"
              << std::endl;
  }

  std::cout << "Mounting " << FLAGS_bench_mds_addr << "/" << FLAGS_bench_fs_name
            << " ..." << std::endl;
  int rc = dingofs_mount(h, FLAGS_bench_mds_addr.c_str(),
                         FLAGS_bench_fs_name.c_str(),
                         FLAGS_bench_mount_point.c_str());
  if (rc != 0) {
    std::cerr << "Mount failed: " << rc << std::endl;
    dingofs_delete(h);
    return 1;
  }

  // Pre-create dir tree serially to avoid MDS TxnWriteConflict
  dingofs_mkdir(h, FLAGS_bench_dir.c_str(), 0755);
  for (int i = 0; i < nthreads; i++) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/%d", FLAGS_bench_dir.c_str(), i);
    dingofs_mkdir(h, dir, 0755);
  }

  size_t total_bytes = static_cast<size_t>(nthreads) * file_size;

  // ---- Phase 1: Prefill ----
  if (!FLAGS_bench_skip_prefill) {
    std::cout << "Phase 1: Prefill " << HumanSize(total_bytes) << " ..."
              << std::endl;
    ProgressTracker prefill_progress(total_bytes, nthreads, "prefill");

    std::vector<std::thread> threads;
    std::vector<int> errors(nthreads, 0);
    double pf_start = NowSec();
    threads.reserve(nthreads);
    for (int i = 0; i < nthreads; i++) {
      threads.emplace_back(PrefillThread, h, i, file_size, block_size,
                           &prefill_progress, &errors[i]);
    }
    for (auto& t : threads) t.join();
    double pf_elapsed = NowSec() - pf_start;
    prefill_progress.Stop();
    if (FLAGS_bench_progress) std::cout << "\r" << std::string(80, ' ') << "\r";

    bool prefill_failed = false;
    for (int i = 0; i < nthreads; i++) {
      if (errors[i] != 0) {
        std::cerr << "Prefill thread " << i << " failed: " << errors[i]
                  << std::endl;
        prefill_failed = true;
      }
    }
    if (prefill_failed) {
      dingofs_umount(h);
      dingofs_delete(h);
      return 1;
    }
    double pf_mb = total_bytes / (1024.0 * 1024.0);
    std::cout << "Prefill done: " << pf_mb << " MB in " << std::fixed
              << std::setprecision(2) << pf_elapsed
              << " s = " << std::setprecision(1) << pf_mb / pf_elapsed
              << " MB/s" << std::endl;

    if (FLAGS_bench_prefill_settle_sec > 0) {
      std::cout << "Settling for " << FLAGS_bench_prefill_settle_sec << " s..."
                << std::endl;
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_bench_prefill_settle_sec));
    }
  }

  // ---- Phase 2: Read ----
  std::cout << "Phase 2: Read (" << FLAGS_bench_pattern << ") ..." << std::endl;
  ProgressTracker read_progress(total_bytes, nthreads, "read");

  std::vector<std::thread> threads;
  std::vector<ThreadResult> results(nthreads);
  double t_start = NowSec();
  threads.reserve(nthreads);
  for (int i = 0; i < nthreads; i++) {
    threads.emplace_back(ReaderThread, h, i, file_size, block_size,
                         random_pattern, &read_progress, &results[i]);
  }
  for (auto& t : threads) t.join();
  double t_end = NowSec();
  read_progress.Stop();
  if (FLAGS_bench_progress) std::cout << "\r" << std::string(80, ' ') << "\r";

  bool has_error = false;
  for (const auto& r : results) {
    if (r.error_code != 0) has_error = true;
  }

  double wall_sec = t_end - t_start;
  double total_data_mb = 0;
  for (const auto& r : results) {
    total_data_mb += r.bytes_read / (1024.0 * 1024.0);
  }

  std::cout << std::endl;
  std::cout << "=== Results (" << FLAGS_bench_pattern
            << " read) ===" << std::endl;
  std::cout << std::left << std::setw(10) << "Thread" << std::right
            << std::setw(12) << "Read" << std::setw(12) << "Elapsed"
            << std::setw(14) << "Throughput" << std::setw(8) << "Status"
            << std::endl;
  std::cout << std::string(56, '-') << std::endl;
  for (const auto& r : results) {
    std::cout << std::left << std::setw(10) << r.thread_id << std::right
              << std::setw(10)
              << static_cast<int>(r.bytes_read / (1024.0 * 1024.0)) << " MB"
              << std::setw(9) << std::fixed << std::setprecision(2)
              << r.elapsed_sec << " s" << std::setw(10) << std::setprecision(1)
              << r.throughput_mbs << " MB/s" << std::setw(8)
              << (r.error_code == 0 ? "  OK" : "  FAIL") << std::endl;
  }
  std::cout << std::string(56, '-') << std::endl;
  std::cout << "Aggregate: " << std::fixed << std::setprecision(1)
            << total_data_mb << " MB in " << std::setprecision(2) << wall_sec
            << " s = " << std::setprecision(1) << total_data_mb / wall_sec
            << " MB/s" << std::endl;

  // ---- Cleanup ----
  if (FLAGS_bench_cleanup) {
    for (int i = 0; i < nthreads; i++) {
      char path[512];
      char dir[512];
      snprintf(path, sizeof(path), "%s/%d/data", FLAGS_bench_dir.c_str(), i);
      snprintf(dir, sizeof(dir), "%s/%d", FLAGS_bench_dir.c_str(), i);
      dingofs_unlink(h, path);
      dingofs_rmdir(h, dir);
    }
    dingofs_rmdir(h, FLAGS_bench_dir.c_str());
  }

  dingofs_umount(h);
  dingofs_delete(h);

  return has_error ? 1 : 0;
}
