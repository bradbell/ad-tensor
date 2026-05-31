// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
/*
-------------------------------------------------------------------------------
{xrst_begin ad_sum dev}

Compute and Record Sum Function
###############################

Prototype
*********
{xrst_literal ,
    BEGIN_SUM, END_SUM
}

Recording
*********
If this thread's tape is recording, and the result is (is not) a constant,
the constant is added to the tape (the operation is added to the tape).

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for left operand, type for sum argument
    start + 1, number of dimensions being summed (n_dim), ad_type_t::none
    start + 2, index of first dimension being summed, ad_type::none
    ...
    start + 1 + n_dim, index of last dimension being summed, ad_type::none

If n_sum is zero, all the dimensions are summed.

where start be the length of arg_value and arg_type before this call to
``ad_t::binary`` .

{xrst_end ad_sum}
*/
// BEGIN_SUM
ad_t ad_t::sum(c10::ArrayRef<long> dim) const
// END_SUM
{
    //
    // res_tensor
    at::Tensor res_tensor = torch::empty( {0} );
    if( dim.size() == 0 ) {
        res_tensor = m_tensor.sum();
    } else {
        res_tensor = m_tensor.sum(dim);
    }
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return ad_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "AD tensor being summed does not match tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = m_ad_type;
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_ad_type == ad_type_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_ad_type == ad_type_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_ad_type == ad_type_t::variable  && "AD tensor being "
                "summed is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::sum );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_ad_type );
        //
        size_t n_sum = dim.size();
        agraph->m_arg_value.push_back( n_sum );
        agraph->m_arg_type.push_back( ad_type_t::none );
        //
        for(size_t i = 0; i < n_sum; ++i) {
            agraph->m_arg_value.push_back( size_t( dim[i] ) );
            agraph->m_arg_type.push_back( ad_type_t::none );
        }
    }
    return ad_t(res_tape_id, res_index, res_tensor, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
