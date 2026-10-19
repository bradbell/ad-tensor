#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <cstddef>
#include <array>
#include <ad_tensor/dev/agraph.hpp>
//
namespace ad_tensor { namespace dev {
    //
    // unpack_call
    std::array<size_t, 5> unpack_call(
        size_t          op_index ,
        const agraph_t& agraph
    );
} }
