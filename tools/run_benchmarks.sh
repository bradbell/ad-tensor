#! /usr/bin/env bash
set -e -u
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# script_path
script_dir="$( dirname -- "${BASH_SOURCE[0]}" )"
script_dir="$( cd -- "$script_dir" &> /dev/null && pwd )"
script_path="$script_dir/$(basename $0)"
# -----------------------------------------------------------------------------
program='build/benchmarks/benchmarks'
$program --gtest_filter=benchmarks.version  | sed -n \
    -e '/^ad-tensor version/p' \
    -e '/^cmake build type/p'
$program --gtest_filter=*:-benchmarks.version  | sed -n \
    -e '/OK *\]/p' \
    -e '/^learn_ms =/p'
# -----------------------------------------------------------------------------
echo "$script_path: OK"
exit 0
