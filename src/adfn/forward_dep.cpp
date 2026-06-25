// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
/*
{xrst_begin adfn_forward_dep usr}

Get The Range Tensors
#####################
Compute the sparsity pattern for how the range values depend
on the domain values.

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_DEP, END_FORWARD_DEP
}

domain_type
***********
This is either ad_type_t::variable or ad_type_t::parameter
and specifies which domain values we are computing the sparsity pattern for.

depend
******
If domain_type in variable (parameter) and (i, j) is in this sparsity pattern,
the value with range index i depends on the
domain variable (parameter) with index j.

{xrst_end adfn_forward_dep}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_FORWARD_DEP
// depend = adfn.forward_dep()
sparsity_t  adfn_t::forward_dep(ad_type_t domain_type) const
{   // END_FORWARD_DEP
    //
    // parameter, variable
    ad_type_t parameter = ad_type_t::parameter;
    ad_type_t variable  = ad_type_t::variable;
    //
    // domain_type, range_type
    dev::user_assert(
        domain_type == parameter || domain_type == variable,
            m_name + ".forward_dep: domain_type is not variable or parameter"
    );
    //
    // sub_sets
    vector<size_t> sub_sets;
    //
    // i_graph
    size_t i_graph_start = 0;
    if( domain_type == variable ) {
        // parameters do not depend on variables so start with variable agraph
        i_graph_start = 1;
    }
    //
    // sparsity
    sparsity_t sparsity;
    for(size_t i_graph = i_graph_start; i_graph < 2; ++i_graph) {
        //
        // range_type agraph
        const dev::agraph_t* agraph = nullptr;
        ad_type_t      range_type;
        if( i_graph == 0 ) {
            range_type = parameter;
            agraph     = &m_par;
        } else {
            range_type = variable;
            agraph     = &m_var;
        }
        //
        // vec_set
        dev::vec_set_t vec_set;
        //
        // n_op, n_dom
        size_t n_op      = agraph->m_op_seq.size();
        for(size_t op_index = 0; op_index < n_op; ++op_index) {
            //
            // op_enum
            dev::op_enum_t op_enum = agraph->m_op_seq[op_index];
            switch( op_enum ) {
                //
                // default
                default: {
                    size_t arg_start = agraph->m_arg_start[op_index];
                    size_t arg_end   = agraph->m_arg_start[op_index + 1];
                    size_t arg_index = arg_start;
                    sub_sets.resize(0);
                    while( arg_index < arg_end )
                    {   if( agraph->m_arg_type[arg_index] == domain_type ) {
                            size_t operand = agraph->m_arg_value[arg_index];
                            sub_sets.push_back( operand);
                        }
                        ++arg_index;
                    }
#ifdef NDEBUG
                    vec_set.union_set(sub_sets);
#else
                    size_t set_id = vec_set.union_set(sub_sets);
                    assert( set_id == op_index );
#endif
                }
                break;
                //
                // dom
                case dev::op_enum_t::dom: {
#ifdef NDEBUG
                    vec_set.singleton_set(op_index);
#else
                    size_t set_id = vec_set.singleton_set(op_index);
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
                dev::call_op_depend(op_index, *agraph, domain_type, vec_set);
                break;
            }
        }
        //
        // sparsity
        for(size_t i = 0; i < m_rng_index.size(); ++i) {
            if( m_rng_ad_type[i] == range_type ) {
                c10::ArrayRef<size_t> set = vec_set.get_set( m_rng_index[i] );
                for(size_t j : set) {
                    std::array<size_t, 2> pair = {i, j};
                    sparsity.push_back(pair);
                }
            }
        }
    }
    return sparsity;
}
} // END_NAMESPACE_AD_TENSOR
