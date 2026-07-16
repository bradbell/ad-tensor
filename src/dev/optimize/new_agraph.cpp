// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/optimize.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
agraph_t new_agraph(
    const agraph_t&       agraph_old,
    bool                  var_op,
    const vector<size_t>& old2new_con,
    const vector<size_t>& old2new_par,
    const vector<size_t>& old2new_var)
{   //
    // not_used
    size_t not_used = std::numeric_limits<size_t>::max();
    //
    // old2new_res
    const vector<size_t>* old2new_res = nullptr;
    if( var_op ) {
        assert( old2new_var.size() == agraph_old.m_op_seq.size() );
        old2new_res = &old2new_var;
    } else {
        assert( old2new_par.size() == agraph_old.m_op_seq.size() );
        old2new_res = &old2new_par;
    }
    //
    // agraph_new
    agraph_t agraph_new;
    size_t n_old = agraph_old.m_op_seq.size();
    size_t op_index_old = 0;
    while(op_index_old < n_old ) {
        size_t op_index_new = (*old2new_res)[op_index_old];
        if( op_index_new != not_used ) {
            op_enum_t op_enum = agraph_old.m_op_seq[op_index_old];
            if(op_enum==op_enum_t::call || op_enum==op_enum_t::call_result) {
                op_index_old = new_call(
                    agraph_new, agraph_old, op_index_old, var_op,
                    old2new_con, old2new_par, old2new_var
                );
            } else {
                assert( op_index_new == agraph_new.m_op_seq.size() );
                //
                // agraph_new: m_op_seq, m_arg_start
                agraph_new.m_op_seq.push_back(
                    agraph_old.m_op_seq[op_index_old]
                );
                agraph_new.m_arg_start.push_back(
                    agraph_new.m_arg_value.size()
                );
                //
                // arg_index
                size_t start = agraph_old.m_arg_start[op_index_old];
                size_t end   = agraph_old.m_arg_start[op_index_old + 1];
                for(size_t arg_index = start; arg_index < end; ++arg_index) {
                    //
                    // agraph_new:: m_arg_type, m_arg_value
                    ad_type_t ad_type = agraph_old.m_arg_type[arg_index];
                    agraph_new.m_arg_type.push_back( ad_type );
                    size_t value_old = agraph_old.m_arg_value[arg_index];
                    size_t value_new = not_used;
                    switch( ad_type ) {
                        //
                        case ad_type_t::constant:
                        value_new = old2new_con[value_old];
                        break;
                        //
                        case ad_type_t::parameter:
                        value_new = old2new_par[value_old];
                        break;
                        //
                        case ad_type_t::variable:
                        value_new = old2new_var[value_old];
                        break;
                        //
                        default:
                        break;
                    }
                    assert( value_new != not_used );
                    agraph_new.m_arg_value.push_back( value_new );
                }
                ++op_index_old;
            }
        }
    }
    return agraph_new;
}
//
} } // END_AD_TENSOR_DEV_NAMESPACE
