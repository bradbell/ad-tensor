#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
namespace ad_tensor { namespace dev {
    //
    // hash_operator
    size_t hash_operator(
        ad_type_t            ad_type,
        const agraph_t&      agraph_old,
        const vector<size_t> old2new,
        size_t               old_index
    );
    //
    // new_op_equal
    bool new_op_equal(
        ad_type_t            ad_type,
        const agraph_t&      agraph_old,
        const vector<size_t> old2new ,
        size_t               old_index_1 ,
        size_t               old_index_2
    );
} }
