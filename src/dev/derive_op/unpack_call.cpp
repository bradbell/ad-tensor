// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <cstddef>
#include <array>
#include <cassert>
#include <ad_tensor/dev/unpack_call.hpp>
/*
{xrst_begin unpack_call dev}
{xrst_spell
    op
    agraph
}

Unpack a Call Operator
######################

Syntax
******
{xrst_code cpp}
    auto [arg_start, atom_id, n_domain, n_range, n_result] = unpack_call(
        op_index, agraph
    );
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_UNPACK_CALL, END_UNPACK_CALL
}

arg_start
*********
The offset in m_arg_value and m_arg_type for this operator.

atom_id
*******
The identifier for this atomic function.

n_domain
********
is the number of domain arguments in this atomic function.

n_range
*******
is the number of range values returned by this atomic function.

n_result
********
is the number of range results that are used.

m_arg_value[ arg_start + 4 + j]
*******************************
is the index corresponding to the j-th domain argument.

m_arg_value[ arg_start + 4 + n_domain + i]
******************************************
is the index in the range vector corresponding to the
i-th result (that is used).

{xrst_end unpack_call}
*/
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// BEGIN_UNPACK_CALL
std::array<size_t, 5> unpack_call(
    size_t          op_index,
    const agraph_t& agraph
)
{   // END_UNPACK_CALL
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t atom_id   = agraph.m_arg_value[arg_start + 0];
    size_t n_domain  = agraph.m_arg_value[arg_start + 1];
    size_t n_range   = agraph.m_arg_value[arg_start + 2];
    size_t n_result  = agraph.m_arg_value[arg_start + 3];
#ifndef NDEBUG
    size_t arg_end   = agraph.m_arg_start[op_index + 1];
    size_t n_arg     = arg_end - arg_start;
    assert( n_arg == 4 + n_domain + n_result);
    for(size_t i = 0; i < 4; ++i) {
        assert( agraph.m_arg_type[arg_start + i] == adtype_t::none );
    }
    for(size_t i = 4; i < 4 + n_domain; ++i) {
        assert( agraph.m_arg_type[arg_start + i] != adtype_t::none );
    }
    for(size_t i = 4 + n_domain; i < 4 + n_domain + n_result; ++i) {
        assert( agraph.m_arg_type[arg_start + i] == adtype_t::none );
    }
# endif
    return std::array<size_t, 5> (
        { arg_start, atom_id, n_domain, n_range, n_result }
    );
}

} } // END_AD_TENSOR_DEV_NAMESPACE
