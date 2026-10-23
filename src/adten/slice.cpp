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
namespace ad_tensor { // Begin ad_tensor::dev
/*
-------------------------------------------------------------------------------
{xrst_begin adten_slice usr}
{xrst_spell
    aslice
}

Slice an AD Tensor in One Dimension
###################################

Syntax
******
{xrst_code cpp}
    aslice = adten.slice(dim, start, end, step)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_SLICE, END_SLICE
}

adten
*****
is the AD tensor that we are slicing.

dim
***
is the dimension that where the slice is taking place

start.value()
*************
is the starting index in the specified dimension.
The starting index is 0 when start does not have a value.

end.value()
***********
is the ending index in the specified dimension.
The slice includes the last index when start does not have a value.

step
****
is the increment between indices that are included in the slice.

aslice
******
is the AD tensor corresponding the slice.

Example
*******
{xrst_literal ,
    examples/adten/slice.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_slice}
-------------------------------------------------------------------------------
{xrst_begin adten_slice_dev dev}

Slice an AD Tensor in One Dimension Developer Documentation
###########################################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_SLICE, END_DEV_SLICE
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for operand,  type of the operand
    start + 1, dim,                adtype::none
    start + 2, start,              adtype::none
    start + 3, end,                adtype::none
    start + 4, step,               adtype::none

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` and n_dim is the number of dimensions in the new shape

{xrst_end adten_slice_dev}
*/
// BEGIN_DEV_SLICE
adten_t adten_t::slice(
    int64_t                dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t                step
) const
// END_DEV_SLICE
{
    //
    // res_tensor
    at::Tensor res_tensor = m_at_ten.slice(dim, start, end, step);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "Tape for AD tensor being sliced is not tape that is recording"
    );
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
                "sliceed is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::slice );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( size_t(dim) );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        if( start.has_value() ) {
            agraph->m_arg_value.push_back( size_t(start.value()) );
        } else {
            agraph->m_arg_value.push_back(0);
        }
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        if( end.has_value() ) {
            agraph->m_arg_value.push_back( size_t(end.value()) );
        } else {
            agraph->m_arg_value.push_back( size_t(m_at_ten.sizes()[dim]) );
        }
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( size_t(step) );
        agraph->m_arg_type.push_back( adtype_t::none );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
