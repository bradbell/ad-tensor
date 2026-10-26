// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_cat usr}
{xrst_spell
    initializer
}

Concatenate Tensors Along One Dimension
#######################################

Syntax
******
{xrst_code cpp}
    result = cat(tensor_list, dim)
{xrst_code}
The syntax above is implemented for both the adten_t and at::Tensor types.
This enables template code, that uses a braced initializer list for tensor_list,
and works for both the at::Tensor and adten_t types.

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_CAT, END_CAT
}

tensor_list
***********
is a list of AD Tensors that are concatenated along the dim dimension.
These tensors must have the same shape except for their size
in the dim dimension.

result
******
AD tensor that results from the concatenation.
The shape of the result is the same as the tensors in the list
expect for the dim dimension which has size equal to the sum of
the size of the dim dimension for the tensors in the list.

Example
*******
{xrst_literal ,
    examples/adten/cat.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_cat}
-------------------------------------------------------------------------------
{xrst_begin adten_cat_dev dev}
{xrst_spell
}

Concatenate Tensors Along One Dimension Developer Documentation
###############################################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_DEV_CAT, END_DEV_CAT
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, dim,                                  none
    start + 1, n_ten: number of tensors in the list, none
    start + 2, index for first tensor,               type of first tensor
    ...,...,...
    start + n_ten + 1, index for last tensor,        type of last tensor

where start is the length of arg_value and arg_type before this call to
``adten_t::cat`` .

{xrst_end adten_cat_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // Begin ad_tensor

// ad_tensor::adten_t::cat
adten_t adten_t::cat(
    const c10::ArrayRef<adten_t>& tensor_list ,
    int64_t                       dim ) {
    //
    // n_ten
    size_t n_ten = tensor_list.size();
    //
    // at_ten_list
    vector<at::Tensor> at_ten_list;
    for(size_t i = 0; i < n_ten; ++i) {
        at_ten_list.push_back( tensor_list[i].m_at_ten );
    }
    //
    // res_at_ten
    at::Tensor res_at_ten = at::cat( at_ten_list, dim);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_at_ten );
    for(size_t i = 0; i < n_ten; ++i) {
        if( tensor_list[i].m_tape_id != tape.m_tape_id ) {
            std::string msg = std::format("cat: tape for tensor_list[{}]", i);
            msg += " is not the tape that is recording";
            dev::user_assert(false, msg);
        }
    }
    //
    // res_adtype
    adtype_t res_adtype = adtype_t::constant;
    for(size_t i = 0; i < n_ten; ++i) {
        res_adtype  = std::max(res_adtype, tensor_list[i].m_adtype);
    }
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
        tape.m_con_vec.push_back( res_at_ten.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::cat );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( size_t(dim) );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( n_ten );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        for(size_t i = 0; i < n_ten; ++i) {
            agraph->m_arg_value.push_back( tensor_list[i].m_index );
            agraph->m_arg_type.push_back( tensor_list[i].m_adtype );
        }
    }
    return adten_t(res_tape_id, res_index, res_at_ten, res_adtype);
}
//
// ad_tensor::cat
adten_t cat(
    const c10::ArrayRef<adten_t>& tensor_list ,
    int64_t                       dim ) {
        return adten_t::cat(tensor_list, dim);
}
at::Tensor cat(
    const c10::ArrayRef<at::Tensor>& tensor_list ,
    int64_t                          dim ) {
        return at::cat(tensor_list, dim);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
