#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/bin/dev_tools.sh !!
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# echo_eval
echo_eval() {
   echo $*
   eval $*
}
# -----------------------------------------------------------------------------
if [ "$0" != "bin/check_all.sh" ]
then
   echo "bin/check_all.sh: must be executed from its parent directory"
   exit 1
fi
if [ "$#" != 0 ] && [ "$#" != 1 ]
then
   echo 'usage: bin/check_all.sh [-skip_check_copy]'
   exit 1
fi
#
# sed
source bin/grep_and_sed.sh
#
# skip_check_copy
skip_check_copy='no'
if [ $# == 1 ]
then
   if [ "$1" == '--skip_check_copy' ]
   then
      skip_check_copy='yes'
   else
      echo 'usage: bin/check_all.sh [-skip_check_copy]'
      exit 1
   fi
fi
# -----------------------------------------------------------------------------
#
# typos
if which typos > /dev/null
then
    echo_eval typos
fi
#
# check_copy.sh
if [ "$skip_check_copy" == 'no' ]
then
   bin/check_copy.sh
fi
#
# check_list
check_list=$(ls bin/check_* | $sed \
   -e '/^bin[/]check_all.sh/d' \
   -e '/^bin[/]check_copy.sh/d' \
)
for check in $check_list
do
   echo_eval $check
done
#
echo "check_all.sh OK"
exit 0
