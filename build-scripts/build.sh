#!/usr/bin/env bash

# Copyright (C) 2021 Jingli Chen (Wine93), NetEase Inc.

############################  GLOBAL VARIABLES

g_stor="fs"
g_list=0
g_depend=0
g_target=""
g_release=0
g_ci=0
g_build_rocksdb=0
g_build_opts=(
    "--define=with_glog=true"
    "--define=libunwind=true"
    "--copt -DHAVE_ZLIB=1"
    "--copt -DGFLAGS_NS=google"
    "--copt -DUSE_BTHREAD_MUTEX"
)

g_os="rocky9"

############################  BASIC FUNCTIONS
get_version() {
    #get tag version
    tag_version=`git status | grep -w "HEAD detached at" | awk '{print $NF}' | awk -F"v" '{print $2}'`

    # get git commit version
    commit_id=`git show --abbrev-commit HEAD|head -n 1|awk '{print $2}'`
    if [ $g_release -eq 1 ]
    then
        debug="+release"
    else
        debug="+debug"
    fi
    if [ -z ${tag_version} ]
    then
        echo "not found version info"
        curve_version=${commit_id}${debug}
    else
        curve_version=${tag_version}+${commit_id}${debug}
    fi
    echo "version: ${curve_version}"
}

msg() {
    printf '%b' "$1" >&2
}

success() {
    msg "\33[32m[✔]\33[0m ${1}${2}"
}

die() {
    msg "\33[31m[✘]\33[0m ${1}${2}"
    exit 1
}

print_title() {
    local delimiter=$(printf '=%.0s' {1..20})
    msg "$delimiter [$1] $delimiter\n"
}

############################ FUNCTIONS
usage () {
    cat << _EOC_
Usage:
    build.sh --stor=bs/fs --list
    build.sh --stor=bs/fs --only=target
Examples:
    build.sh --stor=bs --only=//src/chunkserver:chunkserver
    build.sh --stor=fs --only=src/*
    build.sh --stor=fs --only=test/*
    build.sh --stor=bs --only=test/chunkserver
_EOC_
}

get_options() {
    local args=`getopt -o ldorh --long stor:,list,dep:,only:,os:,release:,ci:,build_rocksdb: -n "$0" -- "$@"`
    eval set -- "${args}"
    while true
    do
        case "$1" in
            -s|--stor)
                g_stor=$2
                shift 2
                ;;
            -l|--list)
                g_list=1
                shift 1
                ;;
            -d|--dep)
                g_depend=$2
                shift 2
                ;;
            -o|--only)
                g_target=$2
                shift 2
                ;;
            -r|--release)
                g_release=$2
                shift 2
                ;;
            -c|--ci)
                g_ci=$2
                shift 2
                ;;
            --os)
                g_os=$2
                shift 2
                ;;
            --build_rocksdb)
                g_build_rocksdb=$2
                shift 2
                ;;
            -h)
                usage
                exit 1
                ;;
            --)
                shift
                break
                ;;
            *)
                exit 1
                ;;
        esac
    done
}

list_target() {
    if [ "$g_stor" == "bs" ]; then
        git submodule update --init -- nbd
        if [ $? -ne 0 ]
        then
            echo "submodule init failed"
            exit
        fi
        print_title " SOURCE TARGETS "
        bazel query 'kind("cc_binary", //src/...)'
        bazel query 'kind("cc_binary", //tools/...)'
        bazel query 'kind("cc_binary", //nebd/src/...)'
        bazel query 'kind("cc_binary", //nbd/src/...)'
        print_title " TEST TARGETS "
        bazel query 'kind("cc_(test|binary)", //test/...)'
        bazel query 'kind("cc_(test|binary)", //nebd/test/...)'
        bazel query 'kind("cc_(test|binary)", //nbd/test/...)'
    elif [ "$g_stor" == "fs" ]; then
        print_title "SOURCE TARGETS"
        bazel query 'kind("cc_binary", //curvefs/src/...)'
        print_title " TEST TARGETS "
        bazel query 'kind("cc_(test|binary)", //curvefs/test/...)'
    fi
}

get_target() {
    if [ "$g_stor" == "bs" ]; then
        bazel query 'kind("cc_(test|binary)", //...)' | grep -E "$g_target" | grep -v "^\/\/curvefs"
    elif [ "$g_stor" == "fs" ]; then
        bazel query 'kind("cc_(test|binary)", //curvefs/...)' | grep -E "$g_target"
    fi
}

build_target() {
    if [ "$g_stor" == "bs" ]; then
        git submodule update --init -- nbd
        if [ $? -ne 0 ]
        then
            echo "submodule init failed"
            exit
        fi
    fi
    local targets
    declare -A result
    if [ $g_release -eq 1 ]; then
        g_build_opts+=("--compilation_mode=opt --copt -g")
        echo "release" > .BUILD_MODE
    else
        g_build_opts+=("--compilation_mode=dbg")
        echo "debug" > .BUILD_MODE
    fi
    g_build_opts+=("--copt -DCURVEVERSION=${curve_version}")

    if [ `gcc -dumpversion | awk -F'.' '{print $1}'` -gt 6 ]; then
        g_build_opts+=("--config=gcc7-later")
    fi

    alltarget=`get_target`
    if [ $g_ci -eq 1 ]; then
        g_build_opts+=("--collect_code_coverage")
        alltarget=...
    fi

    for target in $alltarget
    do
        bazel build ${g_build_opts[@]} $target
        local ret="$?"
        targets+=("$target")
        result["$target"]=$ret
        if [ "$ret" -ne 0 ]; then
            break
        fi
    done

    echo ""
    print_title " BUILD SUMMARY "
    for target in "${targets[@]}"
    do
        if [ "${result[$target]}" -eq 0 ]; then
            success "$target ${version}\n"
        else
            die "$target ${version}\n"
        fi
    done

    # build tools-v2
    g_toolsv2_root="tools-v2"
    if [ $g_release -eq 1 ]
    then
        (cd ${g_toolsv2_root} && make build version=${curve_version})
    else
        (cd ${g_toolsv2_root} && make debug version=${curve_version})
    fi
}


build_requirements() {
    if [[ "$g_stor" == "fs" || $g_ci -eq 1 ]]; then
        git submodule sync && git submodule update --init --recursive
        if [ $? -ne 0 ]; then
            echo "Error: Failed to update git submodules"
            exit 1
        fi
		(cd third-party && rm -rf build installed && cmake -S . -B build && cmake --build build -j 16)
    fi
    g_etcdclient_root="thirdparties/etcdclient"
    (cd ${g_etcdclient_root} && make clean && make all)
    # cp ${g_etcdclient_root}/libetcdclient.h include/etcdclient/etcdclient.h
    # cp ${g_etcdclient_root}/libetcdclient.so include/etcdclient/
}

main() {
    get_options "$@"
    get_version

    if [[ "$g_stor" != "bs" && "$g_stor" != "fs" ]]; then
        die "stor option must be either bs or fs\n"
    fi

    if [ "$g_list" -eq 1 ]; then
        list_target
    elif [[ "$g_target" == "" && "$g_depend" -ne 1 ]]; then
        die "must not disable both only option or dep option\n"
    else
        if [ "$g_depend" -eq 1 ]; then
            build_requirements
        fi
        if [ -n "$g_target" ]; then
            build_target
        fi
    fi
}

############################  MAIN()
main "$@"
