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
/*
-------------------------------------------------------------------------------
{xrst_begin adten_expand usr}
{xrst_spell
    aresult
}

Expand a Tensor in Directions With Size One
###########################################

Syntax
******
{xrst_code cpp}
    aresult = adten.expand(shape)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_EXPAND, END_EXPAND
}

adten
*****
is the AD tensor that we are creating a new expand for.

shape
*****
is the shape for the new expanded tensor.

aresult
*******
is the new expanded AD tensor.

Example
*******
{xrst_literal ,
    examples/adten/expand.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_expand}
-------------------------------------------------------------------------------
{xrst_begin adten_expand_dev dev}
{xrst_spell
}

Expand a Tensor in Directions With Size One Developer Documentation
###################################################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_EXPAND, END_DEV_EXPAND
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for operand,                         type of the operand
    start + 1, number of dimensions in new shape (n_dim), adtype::none
    start + 2, first dimension in new shape,              adtype_t::none
    ..., ..., ...
    start + n_dim + 1, last dimension in new shape,       adtype_t::none

where start is the length of arg_value and arg_type before this call to
``adten_t::expand`` .

{xrst_end adten_expand_dev}
*/
namespace ad_tensor { // Begin ad_tensor::dev
//
// BEGIN_EXPAND BEGIN_DEV_EXPAND
// aresult = adten.expand(shape)
adten_t adten_t::expand(const c10::IntArrayRef& shape) const
// END_EXPAND END_DEV_EXPAND
{
    //
    // res_tensor
    at::Tensor res_tensor = m_at_ten.expand(shape);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "Tape for AD tensor being expanded is not tape that is recording"
    );
    //
    // shape_old, n_dim_old, n_dim_new
    c10::IntArrayRef shape_old = sizes();
    size_t           n_dim_old = shape_old.size();
    size_t           n_dim_new = shape.size();
    //
    // res_adtype
    adtype_t res_adtype = m_adtype;
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_adtype == adtype_t::constant ) {
        // res_index, tape.m_con_vec
        res_index = tape.m_con_vec.size();
        tape.m_con_vec.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "AD tensor being "
                "expanded is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::expand );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( n_dim_new );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        for(size_t i = 0; i < n_dim_new; ++i) {
            agraph->m_arg_type.push_back( adtype_t::none );
            if( i < n_dim_new - n_dim_old ) {
                dev::user_assert( 0 <= shape[i],
                    "expand: a shape element is less than zero and its "
                    "dimension is before dimensions in tensor being expanded"
                );
                agraph->m_arg_value.push_back( size_t( shape[i] ) );
            } else {
                size_t j = i - (n_dim_new - n_dim_old);
                if( shape[i] == -1 ) {
                    agraph->m_arg_value.push_back( size_t( shape_old[j] ) );
                } else {
                    dev::user_assert( 0 <= shape[i],
                        "AD Tensor expand: a shape element is less than zero "
                        "and not -1 for a dimension in tensor being expanded"
                    );
                    agraph->m_arg_value.push_back( size_t( shape[i] ) );
                }
            }
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
