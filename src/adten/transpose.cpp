// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
/*
-------------------------------------------------------------------------------
{xrst_begin adten_transpose usr}
{xrst_spell
    atranspose
}

Create a Transpose of an AD Tensor
##################################

Prototype
*********
{xrst_literal ,
    BEGIN_TRANSPOSE, END_TRANSPOSE
}

adten
*****
is the AD tensor that we are creating a new transpose for.

dim1
****
is the first dimension which we are switching with dim2

dim2
****
is the second dimension which we are switching with dim1

atranspose
**********
is the AD tensor corresponding the transpose
(where dim1 and dim2 have been switched).

{xrst_end adten_transpose}
-------------------------------------------------------------------------------
{xrst_begin adten_transpose_dev dev}

Create a Transpose of an AD Tensor Developer Documentation
##########################################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_TRANSPOSE, END_DEV_TRANSPOSE
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for operand,                         type of the operand
    start + 1, first dimension that we are switching,     ad_type::none
    start + 2, second dimension that we are switching,    ad_type::none

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` and n_dim is the number of dimensions in the new shape

{xrst_end adten_transpose_dev}
*/
// BEGIN_TRANSPOSE BEGIN_DEV_TRANSPOSE
// atranspose = adten.transpose(shape)
adten_t adten_t::transpose(int64_t dim1, int64_t dim2) const
// END_TRANSPOSE END_DEV_TRANSPOSE
{
    //
    // res_tensor
    at::Tensor res_tensor = m_tensor.transpose(dim1, dim2);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "AD tensor being transposeed does not match tape that is recording"
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
                "transposeed is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::transpose );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_ad_type );
        //
        agraph->m_arg_value.push_back( size_t(dim1) );
        agraph->m_arg_type.push_back( ad_type_t::none );
        //
        agraph->m_arg_value.push_back( size_t(dim2) );
        agraph->m_arg_type.push_back( ad_type_t::none );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
