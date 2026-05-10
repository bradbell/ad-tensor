#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/bin/dev_tools.sh !!
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# bin/new_file.sh path_to_file
# Creates a new file with the copyright message at the top.
# If the file name ends with .sh, a bash shebang and sed -e -u are included.
# ----------------------------------------------------------------------------
# path_to_file
if [ "$0" != 'bin/new_file.sh' ]
then
    echo 'new_file.sh must be executed from its parent directory'
    exit 1
fi
if [ $# != 1 ]
then
    echo 'usage: bin/new_file.sh path_to_file'
    exit 1
fi
path_to_file="$1"
if [ -e "$path_to_file" ]
then
    echo "new_file.sh: $path_to_file exists"
    exit 1
fi
if ! echo $path_to_file | grep '[.]' > /dev/null
then
    echo "$path_to_file does not have a file extension"
    exit 1
fi
#
# dir
dir=$(echo $path_to_file | sed -e 's|/[^/]*$||')
if [ "$dir" == "$path_to_file" ]
then
    dir='.'
fi
if [ ! -d "$dir" ]
then
    mkdir -p "$dir"
fi
# -----------------------------------------------------------------------------
# spdx_license_id, spdx_copyright_text, contributor_list
source bin/dev_settings.sh
#
# year
year=$(date +%Y)
#
# fullname
fullname=''
if [ "${USER+x}" != '' ]
then
    for contributor in $contributor_list
    do
        if [[ $contributor == ${USER}* ]]
        then
            fullname=$(echo $contributor | sed -e 's|^.*:||' -e 's|_| |g')
        fi
    done
    if [ "$fullname" == '' ] && [ "${USER+x}" != '' ]
    then
        echo "Cannot user name = $USER in bin/dev_settings.sh contributor_list"
        exit 1
    fi
fi
# ---------------------------------------------------------------------------
#
# ext
ext=$(echo $path_to_file | sed -e 's|^.*[.]\([^.]*\)$|.\1|')
#
# path_to_file
case $ext in

    .sh)
    cat << EOF > $path_to_file
#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/bin/dev_tools.sh !!
# SPDX-License-Identifier: $spdx_license_id
# SPDX-FileCopyrightText: $spdx_copyright_text
# SPDX-FileContributor: $year $fullname
# -----------------------------------------------------------------------------
EOF
    ;;

    .txt)
    cat << EOF > $path_to_file
# SPDX-License-Identifier: $spdx_license_id
# SPDX-FileCopyrightText: $spdx_copyright_text
# SPDX-FileContributor: $year $fullname
# -----------------------------------------------------------------------------
EOF
    ;;

    .hpp|.cpp)
    cat << EOF > $path_to_file
// SPDX-License-Identifier: $spdx_license_id
// SPDX-FileCopyrightText: $spdx_copyright_text
// SPDX-FileContributor: $year $fullname
// ----------------------------------------------------------------------------
EOF
    ;;

    *)
    echo "new_file.sh: $ext is an unknown file extension"
    exit 1

esac
#
echo 'new_file.sh: OK'
exit 0
