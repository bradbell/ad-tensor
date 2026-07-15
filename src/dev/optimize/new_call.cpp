// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/optimize.hpp>
#include <ad_tensor/dev/unpack_call.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
size_t new_call(
    agraph_t&             agraph_new,
    const agraph_t&       agraph_old,
    size_t                op_index_old,
    bool                  var_op,
    const vector<size_t>& old2new_con,
    const vector<size_t>& old2new_par,
    const vector<size_t>& old2new_var) {
    //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_res
    const vector<size_t>* old2new_res = nullptr;
    if( var_op ) {
        old2new_res = &old2new_var;
    } else {
        old2new_res = &old2new_par;
    }
    assert( (*old2new_res)[op_index_old] != not_used );
    //
    // op_index_old
    op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
    while( op_enum != op_enum_t::call ) {
        assert( op_enum == op_enum_t::call_result );
        op_enum = agraph_old.m_op_seq[--op_index_old];
    }
    // arg_start, atom_id, n_domain, n_range, n_result_old
    auto [arg_start, atom_id, n_domain, n_range, n_result_old] = unpack_call(
        op_index_old, agraph_old
    );
    //
    // n_result_new
    size_t n_result_new = 0;
    for(size_t k = 0; k < n_result_old; ++k) {
        if( (*old2new_res)[op_index_old + k] != not_used ) {
            ++n_result_new;
        }
    }
    assert( 0 < n_result_new );
    //
#ifndef NDEBUG
    size_t op_index_new = agraph_new.m_op_seq.size();
    assert( (*old2new_res)[op_index_old] == op_index_new );
#endif
    //
    // agraph_new: m_op_seq, m_arg_start
    agraph_new.m_op_seq.push_back( dev::op_enum_t::call );
    agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
    //
    // agraph_new: m_arg_value, m_arg_type
    agraph_new.m_arg_value.push_back( atom_id );
    agraph_new.m_arg_value.push_back( n_domain );
    agraph_new.m_arg_value.push_back( n_range );
    agraph_new.m_arg_value.push_back( n_result_new );
    for(size_t k = 0; k < 4; ++k) {
        agraph_new.m_arg_type.push_back( ad_type_t::none );
    }
    // agraph_new: m_arg_value, m_arg_type
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index     = arg_start + 4 + j;
        size_t    arg_value_old = agraph_old.m_arg_value[arg_index];
        ad_type_t arg_type      = agraph_old.m_arg_type[arg_index];
        agraph_new.m_arg_type.push_back( arg_type );
        switch( arg_type ) {
            //
            case ad_type_t::constant:
            agraph_new.m_arg_value.push_back( old2new_con[arg_value_old] );
            break;
            //
            case ad_type_t::parameter:
            agraph_new.m_arg_value.push_back( old2new_par[arg_value_old] );
            break;
            //
            case ad_type_t::variable:
            agraph_new.m_arg_value.push_back( old2new_var[arg_value_old] );
            break;
            //
            default:
            assert( false );
        }
    }
    // agraph_new: m_arg_value, m_arg_type
    for(size_t k = 0; k < n_result_old; ++k) {
        if( (*old2new_res)[op_index_old + k] != not_used ) {
            size_t arg_index = arg_start + 4 + n_domain + k;
            size_t arg_value = agraph_old.m_arg_value[ arg_index ];
            agraph_new.m_arg_value.push_back( arg_value );
            agraph_new.m_arg_type.push_back( ad_type_t::none );
        }
    }
    //
    // agraph: m_op_seq, m_arg_start
    for(size_t k = 1; k < n_result_new; ++k) {
        agraph_new.m_op_seq.push_back( dev::op_enum_t::call_result );
        agraph_new.m_arg_start.push_back( agraph_new.m_arg_value.size() );
    }
    //
    return op_index_old + n_result_old;
}
//
} } // END_AD_TENSOR_DEV_NAMESPACE
