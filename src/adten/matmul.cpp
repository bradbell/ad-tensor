// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_matmul usr}
{xrst_spell
    lhs
    rhs
}

Matrix Multiplication
#####################

Prototype
*********
{xrst_literal ,
    BEGIN_MATMUL, END_MATMUL
}

lhs
***
is the left hand side in the matrix multiplication.
This must be one or two dimensional.
If it is one dimensional, it is treated as a row vector.

rhs
***
is the right hand side in the matrix multiplication.
This must be one or two dimensional.
If it is one dimensional, it is treated as a column vector.

prod
****
is the result of the summation.

{xrst_end adten_matmul}
-------------------------------------------------------------------------------
{xrst_begin adten_matmul_dev dev}
{xrst_spell
    lhs
    rhs
}

Compute and Record Matrix Multiplication
########################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_MATMUL, END_DEV_MATMUL
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

    arg_index, arg_value,     arg_type
    start + 0, index for lhs, type for lhs
    start + 1, index for rhs, type for rhs

where start be the length of arg_value and arg_type before this call to
``adten_t::binary`` .

{xrst_end adten_matmul_dev}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR

// BEGIN_MATMUL BEGIN_DEV_MATMUL
// prod = lhs.matmul(rhs)
adten_t adten_t::matmul(const adten_t& rhs) const
// END_MATMUL END_DEV_MATMUL
{
    //
# ifndef NDEBUG
    size_t lhs_n_dim = this->sizes().size();
    size_t rhs_n_dim = rhs.sizes().size();
    //
    dev::user_assert( lhs_n_dim == 1 || lhs_n_dim == 2 ,
        "matmul: lhs is not one or two dimensional"
    );
    dev::user_assert( rhs_n_dim == 1 || rhs_n_dim == 2 ,
        "matmul: rhs is not one or two dimensional"
    );
    //
    int64_t lhs_second_dim;
    if( lhs_n_dim == 1 ) {
        lhs_second_dim = this->sizes()[0];
    } else {
        lhs_second_dim = this->sizes()[1];
    }
    int64_t rhs_first_dim = rhs.sizes()[0];
    dev::user_assert( lhs_second_dim == rhs_first_dim,
        "matmul: lhs times rhs dimension mismatch"
    );
# endif
    //
    // res_tensor
    at::Tensor res_tensor = at::matmul(m_tensor, rhs.m_tensor);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "matmul: "
        "left AD tensor does not match tape that is recording"
    );
    dev::user_assert( rhs.m_tape_id == tape.m_tape_id , "matmul: "
        "right AD tensor does not match tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = std::max( m_ad_type, rhs.m_ad_type );
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
        agraph->m_op_seq.push_back( dev::op_enum_t::matmul );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_ad_type );
        //
        agraph->m_arg_value.push_back( rhs.m_index );
        agraph->m_arg_type.push_back( rhs.m_ad_type );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
