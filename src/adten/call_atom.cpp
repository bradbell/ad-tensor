// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin call_atom usr}

Calling An Atomic Function
##########################
{xrst_spell
    adomain
    arange
}

Prototype
*********
{xrst_literal ,
    // BEGIN_CALL_ATOM, END_CALL_ATOM
}

atom_id
*******
is the atomic function identifier; see :ref:`atom_global@store@atom_id` .

call_info
*********
is the call information passed from the atomic function call
through to its :ref:`base_atom@Virtual Functions` .

adomain
*******
is the AD tensor version of the domain for this atomic function call.

arange
******
is the AD tensor version of the range for this atomic function call.

{xrst_end call_atom}
------------------------------------------------------------------------------
{xrst_begin call_atom_dev dev}
{xrst_spell
    enum
    op
    seq
}

call_atom Developer Documentation
#################################

User Documentation
******************
:ref:`call_atom-name` .

Recording
*********
If this thread's tape is recording:

#.  All the constant results are added to the tape.

#.  If there are parameter results, a callback that computes them
    is added to the tape.

#.  If there are variable results, a callback that computes them
    is added to the tape.

Operation
*********
A callback has the following form in the parameter and variable
acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, atom_id, none
    start + 1, call_info, none
    start + 2, size of domain vector (n_domain), none
    start + 3, size of range vector (n_range), none
    start + 4, number of this graphs's type results (n_result), none
    start + 5, index of first domain argument, argument type
    ..., ..., ...
    start + 4 + n_domain, index of last domain argument, argument type
    start + 5 + n_domain, index in range for first result, ad_type::none
    ..., ..., ...
    start + 4 + n_domain + n_result, index in range for last result, none

The result range result indices are strictly increasing; i.e.,

    arg_value[start + 5 + n_domain] < ... < arg_value[start + 4 + n_result]

op_seq
******
A callback has n_result operations in the tape so that operator
indices are the same as result indices.
The first operation is an ``op_enum_t::call`` and the next
n_result - 1 operations are ``op_enum_t::call_result``.
Only the first operation has entries in arg_value and arg_type.

{xrst_end call_atom_dev}
*/
#include<ad_tensor/vector.hpp>
#include<ad_tensor/adten.hpp>
#include<ad_tensor/atom.hpp>
#include<ad_tensor/base_atom.hpp>
#include<ad_tensor/dev/tape.hpp>
#include<ad_tensor/dev/agraph.hpp>
#include<ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE

// BEGIN_CALL_ATOM
// arange = call_atom(atom_id, call_info, adomain)
vector<adten_t> adten_t::call_atom(
    size_t atom_id, size_t call_info, const vector<adten_t>& adomain
)
{   // END_CALL_ATOM
    //
    // base_atom, long_name
    atom_global_t&      atom_global = atom_global_t::singleton();
    const base_atom_t&  base_atom   = atom_global.get_base_atom(atom_id);
    std::string         long_name   = base_atom.long_name(call_info);
    //
    // n_domain, domain
    size_t n_domain = adomain.size();
    vector<at::Tensor> domain;
    for(size_t j = 0; j < n_domain; ++j) {
        domain.push_back( adomain[j].m_tensor );
    }
    //
    // range, n_range
    vector<bool> rng_used;
    std::optional< vector<at::Tensor> > opt_forward = base_atom.forward(
        call_info, rng_used, domain
    );
    if( ! opt_forward.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".forward did not return a value\n";
        dev::user_assert(false, msg);
    }
    vector<at::Tensor> range = opt_forward.value();
    size_t n_range  = range.size();
    //
    // arange
    vector<adten_t> arange;
    for(size_t i = 0; i < n_range; ++i) {
        arange.push_back( adten_t(range[i]) );
    }
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording ) {
        return arange;
    }
    //
    // pattern
    std::optional<sparsity_t>   opt_depend = base_atom.depend(call_info);
    if( ! opt_depend.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".depend did not return a value\n";
        dev::user_assert(false, msg);
    }
    sparsity_t pattern = opt_depend.value();
    //
    // arange[i].m_ad_type
    for(size_t i = 0; i < n_range; ++i) {
        arange[i].m_ad_type = ad_type_t::constant;
    }
    for(size_t k = 0; k < pattern.size(); ++k) {
        size_t row            = pattern[k][0];
        size_t col            = pattern[k][1];
        ad_type_t ad_type     = arange[row].m_ad_type;
        arange[row].m_ad_type = std::max( ad_type, adomain[col].m_ad_type );
    }
    //
    // tape.m_con, arange[i]: m_index, par_rng_index, var_rng_index
    vector<size_t> par_rng_index, var_rng_index;
    size_t par_index = tape.m_par.m_op_seq.size();
    size_t var_index = tape.m_var.m_op_seq.size();
    for(size_t i = 0; i < n_range; ++i) switch( arange[i].m_ad_type ) {
        //
        // ad_type_t::constant
        case ad_type_t::constant:
        arange[i].m_index = tape.m_con.size();
        tape.m_con.push_back( range[i] );
        break;
        //
        // ad_type_t::parameter
        case ad_type_t::parameter:
        arange[i].m_index = par_index;
        ++par_index;
        par_rng_index.push_back(i);
        break;
        //
        // ad_type_t::variable
        case ad_type_t::variable:
        arange[i].m_index = var_index;
        ++var_index;
        var_rng_index.push_back(i);
        break;
        //
        default:
        assert( false && "call_atom: a domain type is ad_type_t::none");
        break;
    }
    // n_result
    std::array<size_t, 2> n_result = {
        par_index - tape.m_par.m_op_seq.size(),
        var_index - tape.m_var.m_op_seq.size()
    };
    //
    dev::agraph_t*  agraph;
    vector<size_t>* rng_index;
    for(size_t ig = 0; ig < 2; ++ig) if( 0 < n_result[ig] ) {
        //
        if( ig == 0 ) {
            agraph    = &tape.m_par;
            rng_index = &par_rng_index;
        } else {
            agraph    = &tape.m_var;
            rng_index = &var_rng_index;
        }
        assert( n_result[1] == rng_index->size() );
        //
        // agraph: m_op_seq, m_arg_start
        agraph->m_op_seq.push_back( dev::op_enum_t::call );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        // agraph: m_arg_value, m_arg_type
        agraph->m_arg_value.push_back(atom_id);
        agraph->m_arg_value.push_back(call_info);
        agraph->m_arg_value.push_back(n_domain);
        agraph->m_arg_value.push_back(n_range);
        agraph->m_arg_value.push_back(n_result[ig]);
        for(size_t k = 0; k < 5; ++k) {
            agraph->m_arg_type.push_back( ad_type_t::none );
        }
        for(size_t j = 0; j < n_domain; ++j) {
            agraph->m_arg_value.push_back( adomain[j].m_index );
            agraph->m_arg_type.push_back( adomain[j].m_ad_type );
        }
        for(size_t k = 0; k < n_result[ig]; ++k) {
            agraph->m_arg_value.push_back( (*rng_index)[k] );
            agraph->m_arg_type.push_back( ad_type_t::none );
        }
        //
        // agraph: m_op_seq, m_arg_start
        for(size_t k = 1; k < n_result[ig]; ++k) {
            agraph->m_op_seq.push_back( dev::op_enum_t::call_result );
            agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        }
    }
    return arange;
}

} // END_AD_TENSOR_NAMESPACE
