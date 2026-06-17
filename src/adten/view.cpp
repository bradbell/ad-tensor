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
{xrst_begin adten_view usr}
{xrst_spell
    aview
}

Create a View of an AD Tensor
#############################

Prototype
*********
{xrst_literal ,
    BEGIN_VIEW, END_VIEW
}

adten
*****
is the AD tensor that we are creating a new view for.

shape
*****
is the shape for the new view.

aview
*****
is the AD tensor corresponding the new view.

{xrst_end adten_view}
-------------------------------------------------------------------------------
{xrst_begin adten_view_dev dev}

Create a View of an AD Tensor Developer Documentation
#####################################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_VIEW, END_DEV_VIEW
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for operand,                           type of the operand
    start + 1, dim number of dimensions in the new shape,    ad_type::none
    start + 2, index of first m_int64 entry (start_int64), ad_type_t::none

.. csv-table::
    m_int64 index, meaning
    start_int64 + 0, size of first dimension in new shape
    ...
    start_int64 + n_dim - 1, size of last dimension in new shape

where start is the length of arg_value and arg_type before this call to
``adten_t::view`` and n_dim is the number of dimensions in the new shape

{xrst_end adten_view_dev}
*/
// BEGIN_VIEW BEGIN_DEV_VIEW
// aview = adten.view(shape)
adten_t adten_t::view(const c10::IntArrayRef& shape) const
// END_VIEW END_DEV_VIEW
{
    //
    // res_tensor
    at::Tensor res_tensor = m_tensor.view(shape);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "AD tensor being viewed does not match tape that is recording"
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
                "viewed is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::view );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_ad_type );
        //
        size_t n_dim = shape.size();
        agraph->m_arg_value.push_back( n_dim );
        agraph->m_arg_type.push_back( ad_type_t::none );
        //
        size_t start_int64 = agraph->m_int64.size();
        agraph->m_arg_value.push_back( start_int64 );
        agraph->m_arg_type.push_back( ad_type_t::none );
        //
        for(size_t i = 0; i < n_dim; ++i) {
            agraph->m_int64.push_back( shape[i] );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
