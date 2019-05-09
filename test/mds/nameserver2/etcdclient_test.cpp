/*
 * Project: curve
 * Created Date: Saturday March 13th 2019
 * Author: lixiaocui1
 * Copyright (c) 2018 netease
 */

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <thread> //NOLINT
#include <chrono> //NOLINT
#include <cstdlib>
#include <memory>
#include "src/mds/nameserver2/etcd_client.h"
#include "src/mds/nameserver2/namespace_storage.h"
#include "src/common/timeutility.h"
#include "proto/nameserver2.pb.h"

namespace curve {
namespace mds {
// 接口测试
class TestEtcdClinetImp : public ::testing::Test {
 protected:
    TestEtcdClinetImp() {}
    ~TestEtcdClinetImp() {}

    void SetUp() override {
        client_ = std::make_shared<EtcdClientImp>();
        char endpoints[] = "127.0.0.1:2379";
        EtcdConf conf = {endpoints, strlen(endpoints), 20000};
        ASSERT_EQ(0, client_->Init(conf, 200, 3));
        ASSERT_EQ(
            EtcdErrCode::DeadlineExceeded, client_->Put("05", "hello word"));
        ASSERT_EQ(EtcdErrCode::DeadlineExceeded,
            client_->CompareAndSwap("04", "10", "110"));

        client_->SetTimeout(20000);
        system("etcd&");
    }

    void TearDown() override {
        client_ = nullptr;
        system("killall etcd");
        system("rm -fr default.etcd");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

 protected:
    std::shared_ptr<EtcdClientImp> client_;
};

TEST_F(TestEtcdClinetImp, test_EtcdClientInterface) {
    // 1. put file
    // - file0~file9 put到etcd中
    // - file6有快照
    std::map<int, std::string> keyMap;
    std::map<int, std::string> fileName;
    FileInfo fileInfo7, fileInfo8;
    std::string fileInfo9, fileKey10, fileInfo10, fileName10;
    std::string fileInfo6, snapshotKey6, snapshotInfo6, snapshotName6;
    for (int i = 0; i < 11; i++) {
        FileInfo fileinfo;
        std::string filename = "helloword-" + std::to_string(i) + ".log";
        fileinfo.set_id(i);
        fileinfo.set_filename(filename);
        fileinfo.set_parentid(i << 8);
        fileinfo.set_filetype(FileType::INODE_PAGEFILE);
        fileinfo.set_chunksize(DefaultChunkSize);
        fileinfo.set_length(10 << 20);
        fileinfo.set_ctime(::curve::common::TimeUtility::GetTimeofDayUs());
        fileinfo.set_seqnum(1);
        std::string encodeFileInfo;
        ASSERT_TRUE(fileinfo.SerializeToString(&encodeFileInfo));
        std::string encodeKey =
            NameSpaceStorageCodec::EncodeFileStoreKey(i << 8, filename);
        if (i <= 9) {
            ASSERT_EQ(EtcdErrCode::OK,
                client_->Put(encodeKey, encodeFileInfo));
            keyMap[i] = encodeKey;
            fileName[i] = filename;
        }

        if (i == 6) {
            fileinfo.set_seqnum(2);
            ASSERT_TRUE(fileinfo.SerializeToString(&encodeFileInfo));
            fileInfo6 = encodeFileInfo;

            fileinfo.set_seqnum(1);
            snapshotName6 = "helloword-" + std::to_string(i) + ".log.snap";
            fileinfo.set_filename(snapshotName6);
            ASSERT_TRUE(fileinfo.SerializeToString(&snapshotInfo6));
            snapshotKey6 = NameSpaceStorageCodec::EncodeSnapShotFileStoreKey(
                                                                i << 8,
                                                                snapshotName6);
        }

        if (i == 7) {
            fileInfo7.CopyFrom(fileinfo);
        }

        if (i == 8) {
            fileInfo8.CopyFrom(fileinfo);
        }

        if (i == 9) {
            fileInfo9 = encodeFileInfo;
        }

        if (i == 10) {
            fileKey10 = encodeKey;
            fileInfo10 = encodeFileInfo;
            fileName10 = filename;
        }
    }

    // 2. get file, 可以正确获取并解码file0~file9
    for (int i = 0; i < keyMap.size(); i++) {
        std::string out;
        int errCode = client_->Get(keyMap[i], &out);
        ASSERT_EQ(EtcdErrCode::OK, errCode);
        FileInfo fileinfo;
        ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
        ASSERT_EQ(fileName[i], fileinfo.filename());
    }

    // 3. list file, 可以list到file0~file9
    std::vector<std::string> listRes;
    int errCode = client_->List("01", "02", &listRes);
    ASSERT_EQ(EtcdErrCode::OK, errCode);
    ASSERT_EQ(keyMap.size(), listRes.size());
    for (int i = 0; i < listRes.size(); i++) {
        FileInfo finfo;
        ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(listRes[i], &finfo));
        ASSERT_EQ(fileName[i], finfo.filename());
    }

    // 4. delete file, 删除file0~file4，这部分文件不能再获取到
    for (int i = 0; i < keyMap.size()/2; i++) {
        ASSERT_EQ(EtcdErrCode::OK, client_->Delete(keyMap[i]));
        // can not get delete file
        std::string out;
        ASSERT_EQ(
            EtcdErrCode::KeyNotExist, client_->Get(keyMap[i], &out));
    }

    // 5. rename file: rename file9 ~ file10, file10本来不存在
    Operation op1{
        OpType::OpDelete,
        const_cast<char*>(keyMap[9].c_str()),
        const_cast<char*>(fileInfo9.c_str()),
        keyMap[9].size(), fileInfo9.size()};
    Operation op2{
        OpType::OpPut,
         const_cast<char*>(fileKey10.c_str()),
        const_cast<char*>(fileInfo10.c_str()),
        fileKey10.size(), fileInfo10.size()};
    std::vector<Operation> ops{op1, op2};
    ASSERT_EQ(EtcdErrCode::OK, client_->TxnN(ops));
    // cannot get file9
    std::string out;
    ASSERT_EQ(EtcdErrCode::KeyNotExist,
        client_->Get(keyMap[9], &out));
    // get file10 ok
    ASSERT_EQ(EtcdErrCode::OK, client_->Get(fileKey10, &out));
    FileInfo fileinfo;
    ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
    ASSERT_EQ(fileName10, fileinfo.filename());

    // 6. snapshot of keyMap[6]
    Operation op3{
        OpType::OpPut,
        const_cast<char*>(keyMap[6].c_str()),
        const_cast<char*>(fileInfo6.c_str()),
        keyMap[6].size(), fileInfo6.size()};
    Operation op4{
        OpType::OpPut,
        const_cast<char*>(snapshotKey6.c_str()),
        const_cast<char*>(snapshotInfo6.c_str()),
        snapshotKey6.size(), snapshotInfo6.size()};
    ops.clear();
    ops.emplace_back(op3);
    ops.emplace_back(op4);
    ASSERT_EQ(EtcdErrCode::OK, client_->TxnN(ops));
    // get file6
    ASSERT_EQ(EtcdErrCode::OK, client_->Get(keyMap[6], &out));
    ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
    ASSERT_EQ(2, fileinfo.seqnum());
    ASSERT_EQ(fileName[6], fileinfo.filename());
    // get snapshot6
    ASSERT_EQ(EtcdErrCode::OK, client_->Get(snapshotKey6, &out));
    ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
    ASSERT_EQ(1, fileinfo.seqnum());
    ASSERT_EQ(snapshotName6, fileinfo.filename());
    // list snapshotfile
    ASSERT_EQ(EtcdErrCode::OK, client_->List("03", "04", &listRes));
    ASSERT_EQ(1, listRes.size());

    // 7. test CompareAndSwap
    ASSERT_EQ(EtcdErrCode::OK,
        client_->CompareAndSwap("04", "", "100"));
    ASSERT_EQ(EtcdErrCode::OK, client_->Get("04", &out));
    ASSERT_EQ("100", out);

    ASSERT_EQ(EtcdErrCode::OK,
        client_->CompareAndSwap("04", "100", "200"));
    ASSERT_EQ(EtcdErrCode::OK, client_->Get("04", &out));
    ASSERT_EQ("200", out);

    // 8. rename file: rename file7 ~ file8, file8已经存在
    // 把已经存在的file8标记为recycle
    FileInfo recycleFileInfo8;
    recycleFileInfo8.CopyFrom(fileInfo8);
    recycleFileInfo8.set_filestatus(FileStatus::kFileDeleting);
    recycleFileInfo8.set_filetype(INODE_RECYCLE_PAGEFILE);
    std::string encoderecycleFileInfo8Key =
            NameSpaceStorageCodec::EncodeRecycleFileStoreKey(
                recycleFileInfo8.parentid(), recycleFileInfo8.filename());
    std::string encoderecycleFileInfo8;
    ASSERT_TRUE(recycleFileInfo8.SerializeToString(&encoderecycleFileInfo8));
    Operation op7{
        OpType::OpPut,
        const_cast<char*>(encoderecycleFileInfo8Key.c_str()),
        const_cast<char*>(encoderecycleFileInfo8.c_str()),
        encoderecycleFileInfo8Key.size(), encoderecycleFileInfo8.size()};

    // 把file7 rename到 file8
    Operation op8{
        OpType::OpDelete,
        const_cast<char*>(keyMap[7].c_str()), "",
        keyMap[7].size(), 0};
    FileInfo newFileInfo7;
    newFileInfo7.CopyFrom(fileInfo7);
    newFileInfo7.set_parentid(fileInfo8.parentid());
    newFileInfo7.set_filename(fileInfo8.filename());
    std::string encodeNewFileInfo7Key =
            NameSpaceStorageCodec::EncodeFileStoreKey(
                newFileInfo7.parentid(), newFileInfo7.filename());
    std::string encodeNewFileInfo7;
    ASSERT_TRUE(newFileInfo7.SerializeToString(&encodeNewFileInfo7));
    Operation op9{
        OpType::OpPut,
        const_cast<char*>(encodeNewFileInfo7Key.c_str()),
        const_cast<char*>(encodeNewFileInfo7.c_str()),
        encodeNewFileInfo7Key.size(), encodeNewFileInfo7.size()};
    ops.clear();
    ops.emplace_back(op7);
    ops.emplace_back(op8);
    ops.emplace_back(op9);
    ASSERT_EQ(EtcdErrCode::OK, client_->TxnN(ops));
    // 不能获取 file7
    ASSERT_EQ(EtcdErrCode::KeyNotExist,
        client_->Get(keyMap[7], &out));
    // 成功获取rename以后的file7
    ASSERT_EQ(EtcdErrCode::OK, client_->Get(keyMap[8], &out));
    ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
    ASSERT_EQ(newFileInfo7.filename(), fileinfo.filename());
    ASSERT_EQ(newFileInfo7.filetype(), fileinfo.filetype());
    // 成功获取recycle状态的file8
    ASSERT_EQ(EtcdErrCode::OK, client_->Get(encoderecycleFileInfo8Key, &out));
    ASSERT_TRUE(NameSpaceStorageCodec::DecodeFileInfo(out, &fileinfo));
    ASSERT_EQ(FileStatus::kFileDeleting, fileinfo.filestatus());
    ASSERT_EQ(FileType::INODE_RECYCLE_PAGEFILE, fileinfo.filetype());

    // 9. test more Txn err
    ops.emplace_back(op9);
    ASSERT_EQ(EtcdErrCode::InvalidArgument, client_->TxnN(ops));

    // 10. abnormal
    ops.clear();
    ops.emplace_back(op3);
    ops.emplace_back(op4);
    client_->SetTimeout(0);
    ASSERT_EQ(
        EtcdErrCode::DeadlineExceeded, client_->Put(fileKey10, fileInfo10));
    ASSERT_EQ(EtcdErrCode::DeadlineExceeded, client_->Delete("05"));
    ASSERT_EQ(EtcdErrCode::DeadlineExceeded, client_->Get(fileKey10, &out));
    ASSERT_EQ(EtcdErrCode::DeadlineExceeded, client_->TxnN(ops));

    client_->SetTimeout(5000);
    Operation op5{
        OpType(5), const_cast<char*>(snapshotKey6.c_str()),
        const_cast<char*>(snapshotInfo6.c_str()),
        snapshotKey6.size(), snapshotInfo6.size()};
    ops.clear();
    ops.emplace_back(op3);
    ops.emplace_back(op5);
    ASSERT_EQ(EtcdErrCode::TxnUnkownOp, client_->TxnN(ops));

    // close client
    client_->CloseClient();
    ASSERT_EQ(EtcdErrCode::Canceled, client_->Put(fileKey10, fileInfo10));
    ASSERT_FALSE(
        EtcdErrCode::OK == client_->CompareAndSwap("04", "300", "400"));
}
}  // namespace mds
}  // namespace curve