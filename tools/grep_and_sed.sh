# ---------------------------------------------------------------------------
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# ---------------------------------------------------------------------------
# source tools/grep_and_sed
# Sets the shell variables grep and sed to the gnu version for this system
# ---------------------------------------------------------------------------
#
# sed, grep
sed=sed
grep=grep
if [ "$(uname)" == 'Darwin' ]
then
if which gsed > /dev/null
    then
        sed=gsed
    elif [ -e /opt/homebrew/bin/gsed ]
    then
       sed=/opt/homebrew/bin/gsed
    else
        echo 'darwin_names.sh: cannot find gsed (gnu sed) on MacOS system'
    fi
    if which ggrep > /dev/null
    then
        grep=ggrep
    elif [ -e /opt/homebrew/bin/ggrep ]
    then
       grep=/opt/homebrew/bin/ggrep
    else
        echo 'darwin_names.sh: cannot find ggrep (gnu grep) on MacOS system'
    fi
fi
