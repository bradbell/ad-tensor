// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/dev/vec_sets.hpp>
#include <ad_tensor/dev/to_string.hpp>
/*
{xrst_begin adfn_forward_dep usr}

Determine Range Dependencies
############################
Compute the sparsity pattern for how the range values depend
on the domain values.

Syntax
******
{xrst_code cpp}
[depend_par, depend_var] = adfn.forward_dep()
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adfn.hpp
    BEGIN_FORWARD_DEP, END_FORWARD_DEP
}

trace
*****
if :ref:`adfn@trace` is true, this calculation will be traced.

depend_par
**********
If the index (i, j) is in depend_par,
the value at range index i depends on the domain parameter with index j,

depend_var
**********
If the index (i, j) is in depend_var,
the value at range index i value depends on the domain variable with index j,

{xrst_end adfn_forward_dep}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
std::tuple<sparsity_t, sparsity_t> adfn_t::forward_dep(void) const
{   //
    // cout, to_string
    using std::cout;
    using ad_tensor::dev::to_string;
    //
    // parameter, variable
    ad_type_t parameter = ad_type_t::parameter;
    ad_type_t variable  = ad_type_t::variable;
    //
    // sub_sets
    vector<size_t> sub_sets;
    //
    // vec_sets
    dev::vec_sets_t vec_sets;
    //
    // trace
    bool trace = m_options.get_trace();
    if( trace ) {
        cout << "Begin tracing " + get_name() + ".forward_dep\n";
    }
    //
    // ------------------------------------------------------------------------
    // n_par, op_index, agraph, vec_sets
    size_t n_par = m_par.m_op_seq.size();
    for(size_t op_index = 0; op_index < n_par; ++op_index) {
        const dev::agraph_t& agraph = m_par;
        //
        // op_enum
        dev::op_enum_t op_enum = agraph.m_op_seq[op_index];
        switch( op_enum ) {
            //
            // default
            default: {
                size_t arg_start = agraph.m_arg_start[op_index];
                size_t arg_end   = agraph.m_arg_start[op_index + 1];
                sub_sets.resize(0);
                for(size_t arg_index = arg_start; arg_index < arg_end;
                ++arg_index) {
                    ad_type_t domain_type = agraph.m_arg_type[arg_index];
                    if( domain_type == parameter ) {
                        size_t operand = agraph.m_arg_value[arg_index];
                        sub_sets.push_back( operand);
                    }
                }
#ifdef NDEBUG
                vec_sets.union_set(sub_sets);
#else
                size_t set_id = vec_sets.union_set(sub_sets);
                assert( set_id == op_index );
#endif
            }
            break;
            //
            // dom
            case dev::op_enum_t::dom: {
#ifdef NDEBUG
                vec_sets.singleton_set(op_index);
#else
                size_t set_id = vec_sets.singleton_set(op_index);
                assert( set_id == op_index );
#endif
            }
            break;
            //
            // call_result
            case dev::op_enum_t::call_result:
            break;
            //
            // call
            case dev::op_enum_t::call:
            dev::call_op_depend(op_index, agraph, n_par, vec_sets);
            break;
        }
        if( trace ) {
            const c10::ArrayRef<size_t> set = vec_sets.get_set(op_index);
            cout << "set(" << op_index << ") = " << to_string(set) << ", ";
            cout << dev::to_string(op_enum)  << "(";
            size_t start = m_par.m_arg_start[op_index];
            size_t stop  = m_par.m_arg_start[op_index + 1];
            for(size_t i = start; i < stop; ++i) {
                cout << "[" << m_par.m_arg_value[i] << ",";
                cout << dev::to_string( m_par.m_arg_type[i] ) << "]";
            }
            cout << ")\n";
        }
    }
    if( trace ) {
        cout << "n_par = " << n_par << "\n";
    }
    // ------------------------------------------------------------------------
    // op_index, agraph, vec_sets
    size_t n_var = m_var.m_op_seq.size();
    for(size_t op_index = 0; op_index < n_var; ++op_index) {
        const dev::agraph_t& agraph = m_var;
        //
        // op_enum
        dev::op_enum_t op_enum = agraph.m_op_seq[op_index];
        switch( op_enum ) {
            //
            // default
            default: {
                size_t arg_start = agraph.m_arg_start[op_index];
                size_t arg_end   = agraph.m_arg_start[op_index + 1];
                sub_sets.resize(0);
                for(size_t arg_index = arg_start; arg_index < arg_end;
                ++arg_index) {
                    ad_type_t domain_type = agraph.m_arg_type[arg_index];
                    if( domain_type == parameter ) {
                        size_t operand = agraph.m_arg_value[arg_index];
                        sub_sets.push_back( operand);
                    } else if ( domain_type == variable ) {
                        size_t operand = agraph.m_arg_value[arg_index];
                        sub_sets.push_back(operand + n_par);
                    }
                }
#ifdef NDEBUG
                vec_sets.union_set(sub_sets);
#else
                size_t set_id = vec_sets.union_set(sub_sets);
                assert( set_id == op_index + n_par);
#endif
            }
            break;
            //
            // dom
            case dev::op_enum_t::dom: {
#ifdef NDEBUG
                vec_sets.singleton_set(op_index + n_par);
#else
                size_t set_id = vec_sets.singleton_set(op_index + n_par);
                assert( set_id == op_index + n_par);
#endif
            }
            break;
            //
            // call_result
            case dev::op_enum_t::call_result:
            break;
            //
            // call
            case dev::op_enum_t::call:
            dev::call_op_depend( op_index, agraph, n_par, vec_sets);
            break;
        }
        if( trace ) {
            size_t set_id = op_index + n_par;
            const c10::ArrayRef<size_t> set = vec_sets.get_set(set_id);
            cout << "set(" << set_id << ") = " << to_string(set) << ", ";
            cout << dev::to_string(op_enum)  << "(";
            size_t start = m_var.m_arg_start[op_index];
            size_t stop  = m_var.m_arg_start[op_index + 1];
            for(size_t i = start; i < stop; ++i) {
                cout << "[" << m_var.m_arg_value[i] << ",";
                cout << dev::to_string( m_var.m_arg_type[i] ) << "]";
            }
            cout << ")\n";
        }
    }
    //
    // depend_par
    sparsity_t depend_par;
    for(size_t i = 0; i < m_rng_index.size(); ++i) {
        ad_type_t range_type = m_rng_ad_type[i];
        if( range_type == parameter ) {
            size_t rng_set_id = m_rng_index[i];
            c10::ArrayRef<size_t> set = vec_sets.get_set( rng_set_id );
            for(size_t j : set) {
                std::array<size_t, 2> pair = {i, j};
                depend_par.push_back(pair);
            }
        }  else if( range_type == variable ) {
            size_t rng_set_id = m_rng_index[i] + n_par;
            c10::ArrayRef<size_t> set = vec_sets.get_set( rng_set_id );
            for(size_t dom_set_id : set) {
                if( dom_set_id < n_par ) {
                    size_t j = dom_set_id;
                    std::array<size_t, 2> pair = {i, j};
                    depend_par.push_back(pair);
                }
            }
        }
    }
    //
    // depend_var
    sparsity_t depend_var;
    for(size_t i = 0; i < m_rng_index.size(); ++i) {
        ad_type_t range_type = m_rng_ad_type[i];
        if( range_type == variable ) {
            size_t rng_set_id = m_rng_index[i] + n_par;
            c10::ArrayRef<size_t> set = vec_sets.get_set( rng_set_id );
            for(size_t dom_set_id : set) {
                if( n_par <= dom_set_id ) {
                    size_t j = dom_set_id - n_par;
                    std::array<size_t, 2> pair = {i, j};
                    depend_var.push_back(pair);
                }
            }
        }
    }
    if( trace ) {
        cout << "End tracing " + get_name() + ".forward_dep\n";
    }
    return std::tuple<sparsity_t, sparsity_t>(depend_par, depend_var);
}
} // END_NAMESPACE_AD_TENSOR
