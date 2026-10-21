#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/tools/dev_tools.sh !!
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# echo_eval
echo_eval() {
   echo $*
   eval $*
}
#
# script_path
script_dir="$( dirname -- "${BASH_SOURCE[0]}" )"
script_dir="$( cd -- "$script_dir" &> /dev/null && pwd )"
script_path="$script_dir/$(basename $0)"
# -----------------------------------------------------------------------------
if [ ! -e 'tools/n_compile_job.sh' ]
then
   echo "tools/n_compile_job.sh: must be executed from its parent directory"
   exit 1
fi
#
# n_proc
if [ $(uname) != Darwin ]
then
   n_proc=$(nproc)
else
   n_proc=$(sysctl -n hw.ncpu)
fi
#
# max_job
if [ $(uname) != Darwin ]
then
    # make sure free reports 1GB of available memory for each job
    if [ $( free -g | awk '{ if ( NR == 1 ) { print $6} }') != available ]
    then
        echo 'check_gtest.sh: Unexpected column order in output of free'
        exit 1
    fi
    max_jobs=$( free -g | awk '{ if ($1 == "Mem:") { print $7 } }' )
else
    max_jobs=$( vm_stat | awk -v pagesize=$(pagesize) '
        /Pages free/ {free=$3}
        /Pages inactive/ {inactive=$3}
        /Pages speculative/ {speculative=$3}
        /Pages purgeable/ {purgeable=$3}
        END {
            # pages to GB conversion factor
            to_gb = pagesize / (1024. * 1024. * 1024.);
            #
            print int( (free + inactive + speculative + purgeable) * to_gb );
        }
    ')
fi
#
# n_job
if (( n_proc > max_jobs ))
then
    n_job=$max_jobs
else
    n_job=$n_proc
fi
if (( n_job < 1 ))
then
    echo 'n_compile_job.sh: Note enough available memory reported.'
    echo 'Need one GB for each compile process'
    exit 1
fi
echo $n_job
