#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <cstddef>
#include <tuple>
#include <ad_tensor/dev/agraph.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // unpack_call
    inline std::tuple<size_t, size_t, size_t, size_t, size_t> unpack_call(
        size_t         op_index,
        const agraph_t agraph  ) {
        size_t arg_start = agraph.m_arg_start[op_index];
        size_t atom_id   = agraph.m_arg_value[arg_start + 0];
        size_t n_domain  = agraph.m_arg_value[arg_start + 1];
        size_t n_range   = agraph.m_arg_value[arg_start + 2];
        size_t n_result  = agraph.m_arg_value[arg_start + 3];
        return std::tuple<size_t, size_t, size_t, size_t, size_t> (
            arg_start, atom_id, n_domain, n_range, n_result
        );
    }
} }
