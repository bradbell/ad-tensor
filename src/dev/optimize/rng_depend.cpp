// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// atom_depend
void atom_depend(
    size_t          op_index    ,
    bool            var_op      ,
    const agraph_t& agraph      ,
    vector<bool>&   depend_con  ,
    vector<bool>&   depend_par  ,
    vector<bool>&   depend_var  ) {
    //
    assert( agraph.m_op_seq[op_index] == op_enum_t::call );
    //
    // arg_start, n_domain, n_range, n_result
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t atom_id   = agraph.m_arg_value[arg_start + 0];
    size_t call_info = agraph.m_arg_value[arg_start + 1];
    size_t n_result  = agraph.m_arg_value[arg_start + 4];
    //
    // base_atom, long_name
    dev::atom_global_t& atom_global = dev::atom_global_t::singleton();
    const base_atom_t&  base_atom   = atom_global.get_base_atom(atom_id);
    std::string         long_name   = base_atom.long_name(call_info);
    //
    // sparsity
    std::optional<sparsity_t> opt = base_atom.depend(call_info);
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".depend did not return a value\n";
        dev::user_assert(false, msg);
    }
    sparsity_t sparsity = opt.value();
    sparsity.sort();
    //
    // depend_con, depend_par, depend_var
    size_t r_index = 0;
    size_t s_index = 0;
    for( r_index = 0; r_index < n_result; ++r_index) {
        //
        // spa_index
        while(s_index < sparsity.size() && sparsity[s_index][0] < r_index ) {
            ++s_index;
        }
        while(s_index < sparsity.size() && sparsity[s_index][0] == r_index ) {
            size_t d_index    = sparsity[s_index][1];
            size_t value      = agraph.m_arg_value[arg_start + 5 + d_index];
            ad_type_t ad_type = agraph.m_arg_type[arg_start + 5 + d_index];
            switch( ad_type ) {
                //
                case ad_type_t::constant:
                depend_con[value] = true;
                break;
                //
                case ad_type_t::parameter:
                depend_par[value] = true;
                break;
                //
                case ad_type_t::variable:
                assert( var_op );
                depend_var[value] = true;
                break;
                //
                default:
                assert(false);
            }
            ++s_index;
        }
    }
}
//
// op_depend
void op_depend(
    bool            var_op      ,
    const agraph_t& agraph      ,
    vector<bool>&   depend_con  ,
    vector<bool>&   depend_par  ,
    vector<bool>&   depend_var  ) {
    //
    // res_depend
    auto res_depend = [&](size_t op_index) {
        if( var_op ) {
            return depend_var[op_index];
        }
        return depend_par[op_index];
    };
    //
    // is_call_op
    auto is_call_op = [&](size_t op_index) {
        op_enum_t op_enum = agraph.m_op_seq[op_index];
        return op_enum == op_enum_t::call || op_enum == op_enum_t::call_result;
    };
    //
    // n_op
    size_t n_op = agraph.m_op_seq.size();
    //
    // op_index
    size_t op_index = n_op;
    while( 0 < op_index ) {
        --op_index;
        //
        if( is_call_op(op_index) ) {
            op_enum_t op_enum = agraph.m_op_seq[op_index];
            while( op_enum == op_enum_t::call_result ) {
                --op_index;
            }
            atom_depend(
                op_index, var_op, agraph, depend_con, depend_par, depend_var
            );
        } else if( res_depend(op_index) ) {
            size_t start = agraph.m_arg_start[op_index];
            size_t end   = agraph.m_arg_start[op_index + 1];
            for(size_t arg_index = start; arg_index < end; ++arg_index) {
                size_t value = agraph.m_arg_value[arg_index];
                switch( agraph.m_arg_type[arg_index] ) {
                    //
                    case ad_type_t::constant:
                    depend_con[value] = true;
                    break;
                    //
                    case ad_type_t::parameter:
                    depend_par[value] = true;
                    break;
                    //
                    case ad_type_t::variable:
                    assert( var_op );
                    depend_var[value] = true;
                    break;
                    //
                    default:
                    assert(false);
                }
            }
        }
    }
    return;
}
//
// rng_depend
std::tuple< vector<bool>, vector<bool>, vector<bool> >
rng_depend(const adfn_t* adfn) {
    return adfn->rng_depend();
}
} } // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
namespace ad_tensor {
    //
    // adfn::depend
    std::tuple< vector<bool>, vector<bool>, vector<bool> > adfn_t::rng_depend(
        void
    ) const {
        //
        // depend_con, depend_par, depend_var
        vector<bool> depend_con, depend_par, depend_var;
        //
        // n_con, n_par, n_var, n_rng
        size_t n_con = m_con.size();
        size_t n_par = m_par.m_op_seq.size();
        size_t n_var = m_var.m_op_seq.size();
        size_t n_rng = m_rng_index.size();
        //
        // depend_con, depend_par, depend_var
        depend_con.resize(n_con, false);
        depend_par.resize(n_par, false);
        depend_var.resize(n_var, false);
        //
        // depend_con, depend_par, depend_var
        for(size_t i = 0; i < n_rng; ++i) { switch( m_rng_ad_type[i] ) {
            //
            case ad_type_t::constant:
            depend_con[ m_rng_index[i] ] = true;
            break;
            //
            case ad_type_t::parameter:
            depend_par[ m_rng_index[i] ] = true;
            break;
            //
            case ad_type_t::variable:
            depend_var[ m_rng_index[i] ] = true;
            break;
            //
            default:
            assert(false);
        } }
        //
        // depend_con, depend_par, depend_var
        bool var_op = true;
        op_depend(var_op, m_var, depend_con, depend_par, depend_var);
        //
        // depend_con, depend_par
        var_op = false;
        op_depend(var_op, m_par, depend_con, depend_par, depend_var);
        //
        return std::tuple< vector<bool>, vector<bool>, vector<bool> > (
            depend_con, depend_par, depend_var
        );
    }
}
