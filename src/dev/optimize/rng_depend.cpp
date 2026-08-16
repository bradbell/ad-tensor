// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin rng_depend dev}
{xrst_spell
    dev
    adfn
}

Determine Which Values the Range Depends on
###########################################

Syntax
******
{xrst_code cpp}
auto [depend_con, depend_par, depend_var] = dev::rng_depend(adfn)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_RNG_DEPEND, END_RNG_DEPEND
}

adfn
****
is the AD function object we are computing the dependencies for.

depend_con
**********
The length of this vector is the number of constants stored in adfn.
If depend_con[k] is true (false), the adfn :ref:`adfn_get_range@range`
depends (does not depend) in the k-th constant in adfn.

depend_par
**********
The length of this vector is the number of parameter in
:ref:`adfn_forward_par@par_all` for this adfn.
If depend_par[k] is true (false), the range
depends (does not depend) in the k-th parameter in par_all.

depend_var
**********
The length of this vector is the number of variables in
:ref:`adfn_forward_var@var_all` for this adfn.
If depend_var[k] is true (false), the range
depends (does not depend) in the k-th variable in var_all.

{xrst_end rng_depend}
*/
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/dev/unpack_call.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// atom_depend
void atom_depend(
    size_t          call_op_index    ,
    bool            var_op           ,
    const agraph_t& agraph           ,
    vector<bool>&   depend_con       ,
    vector<bool>&   depend_par       ,
    vector<bool>&   depend_var       ) {
    //
    assert( agraph.m_op_seq[call_op_index] == op_enum_t::call );
    //
    // res_depend
    auto res_depend = [&](size_t op_index) {
        if( var_op ) {
            return depend_var[op_index];
        }
        return depend_par[op_index];
    };
    //
    // arg_start, atom-id, n_result
    auto [arg_start, atom_id, n_domain, n_range, n_result] = unpack_call(
        call_op_index, agraph
    );
    //
    // base_atom
    dev::atom_global_t& atom_global = dev::atom_global_t::singleton();
    const base_atom_t&  base_atom   = atom_global.get_base_atom(atom_id);
    //
    // pattern
    std::optional<sparsity_t> opt = base_atom.depend();
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + base_atom.get_name();
        msg += ".depend did not return a value\n";
        dev::user_assert(false, msg);
    }
    sparsity_t pattern = opt.value();
    pattern.sort();
    //
    // depend_con, depend_par, depend_var
    size_t s_index = 0;
    for(size_t res_index = 0; res_index < n_result; ++res_index) {
    if( res_depend(call_op_index + res_index) ) {
        //
        // rng_index
        // This assumes that rng_index is increasing w.r.t. res_index
        size_t arg_index = arg_start + 4 + n_domain + res_index;
        size_t rng_index = agraph.m_arg_value[arg_index];
        //
        // s_index
        while(s_index < pattern.size() && pattern[s_index][0] < rng_index ) {
            ++s_index;
        }
        while(s_index < pattern.size() && pattern[s_index][0] == rng_index ) {
            size_t d_index    = pattern[s_index][1];
            size_t value      = agraph.m_arg_value[arg_start + 4 + d_index];
            adtype_t adtype = agraph.m_arg_type[arg_start + 4 + d_index];
            switch( adtype ) {
                //
                case adtype_t::constant:
                depend_con[value] = true;
                break;
                //
                case adtype_t::parameter:
                depend_par[value] = true;
                break;
                //
                case adtype_t::variable:
                assert( var_op );
                depend_var[value] = true;
                break;
                //
                default:
                assert(false);
            }
            ++s_index;
        }
    } }
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
                assert( 0 < op_index );
                --op_index;
                op_enum = agraph.m_op_seq[op_index];
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
                    case adtype_t::constant:
                    depend_con[value] = true;
                    break;
                    //
                    case adtype_t::parameter:
                    depend_par[value] = true;
                    break;
                    //
                    case adtype_t::variable:
                    assert( var_op );
                    depend_var[value] = true;
                    break;
                    //
                    default:
                    break;
                }
            }
        }
    }
    return;
}
//
// BEGIN_RNG_DEPEND
std::array< vector<bool>, 3 >
rng_depend(const adfn_t* adfn)
{   // END_RNG_DEPEND
    return adfn->rng_depend();
}
} } // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
namespace ad_tensor {
    //
    // adfn::depend
    std::array< vector<bool>, 3 > adfn_t::rng_depend(
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
        for(size_t i = 0; i < n_rng; ++i) { switch( m_rng_adtype[i] ) {
            //
            case adtype_t::constant:
            depend_con[ m_rng_index[i] ] = true;
            break;
            //
            case adtype_t::parameter:
            depend_par[ m_rng_index[i] ] = true;
            break;
            //
            case adtype_t::variable:
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
        return std::array< vector<bool>, 3 > (
            { depend_con, depend_par, depend_var }
        );
    }
}
