// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_index usr}
{xrst_spell
}

Extracting Some Elements from a Tensor
######################################

Syntax
******
{xrst_code cpp}
    extract = from.index(index_list)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_INDEX, END_INDEX
}

from
****
is the tensor we are extracting elements from.
We use m for the number of dimensions in tensor; i.e.

    m = from.sizes().size();

index_list
**********
Is the list of indices that specify which elements are extracted.
There must be m tensors in index_list and
each tensor in this list must be one dimensional and
have the same number of elements which we denote by n.

extract
*******
is a size n one dimensional tensor of extracted values.
Let index_0, ... , index_m1 denote corresponding elements of index_list.
The value of extract is described by the following pseudo code:
{xrst_code cpp}
    for(j = 0; j < n; ++j) {
        extract[j] = from[ index_0[j], ... , index_m1[j] ]
    }
{xrst_code}

Example
*******
{xrst_literal ,
    examples/adten/index.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_index}
-------------------------------------------------------------------------------
{xrst_begin adten_index_dev dev}
{xrst_spell
    idx
}

Compute and Record an Element Extract Operation
###############################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_INDEX, END_DEV_INDEX
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
    start + 0, index for from tensor,                     type of from
    start + 1, number of indices in index_list  (n_idx),  adtype_t::none
    start + 2, index  of tensor index_list[0]             adtype_t::constant
    ..., ..., ...
    start + 1 + n_idx, index of index_list[n_idx-1]       adtype_t::constant

where start is the length of arg_value and arg_type before this call to
``adten_t::binary``,
If n_dim is zero, all the dimensions are summed.


{xrst_end adten_index_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_INDEX  BEGIN_DEV_INDEX
adten_t adten_t::index(
        const c10::List< std::optional<at::Tensor> >& index_list
    ) const
{   // END_INDEX END_DEV_INDEX
    //
    // before
    const adten_t& from = *this;
    //
    // res_tensor
    at::Tensor res_tensor = from.m_at_ten.index(index_list);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    //
    dev::user_assert( from.m_tape_id == tape.m_tape_id ,
        "index: Tape for from is not tape that is recording"
    );
    dev::user_assert( index_list.size() == from.sizes().size(),
        "index: index_list.size() not equal number of dimensions in from"
    );
    //
    // res_adtype
    adtype_t res_adtype = from.m_adtype;
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_adtype == adtype_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "index: "
                "before or replace is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::index );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        // start + 0
        agraph->m_arg_value.push_back( from.m_index );
        agraph->m_arg_type.push_back( from.m_adtype );
        //
        // start + 1
        size_t n_idx = index_list.size();
        agraph->m_arg_value.push_back( n_idx );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        // start + 2 + i
        for (const std::optional<at::Tensor>& index : index_list) {
            size_t con_index  = tape.m_con.size();
            tape.m_con.push_back( index.value().clone() );
            agraph->m_arg_value.push_back( con_index );
            agraph->m_arg_type.push_back( adtype_t::constant );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
