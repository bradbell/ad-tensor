# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#[[
{xrst_begin benchmarks_lib ben}

The Benchmarks Library
######################
General purpose routines used by the ad_tensor benchmarks:
{xrst_toc_table
    benchmarks/lib/elapsed_ms.cpp
}

{xrst_end benchmarks_lib}
]]
#
add_library(benchmarks_lib SHARED
    elapsed_ms.cpp
)
target_link_libraries(benchmarks_lib
    PRIVATE
    ${TORCH_LIBRARIES}
)
