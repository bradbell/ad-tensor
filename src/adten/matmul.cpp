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

Syntax
******
{xrst_code cpp}
    prod = lhs.matmul(rhs)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_MATMUL, END_MATMUL
}

lhs
***
is the left hand side in the matrix multiplication.
It must be two or more dimensional.
If it is more than two dimensional; i.e. sizes (\*,m,n), broadcasting is used
to match rhs for the \* dimensions.

rhs
***
is the right hand side in the matrix multiplication.
It must be two or more dimensional.
If it is more than two dimensional; i.e. sizes (\*,n,p), broadcasting is used
to match lhs for the \* dimensions.

prod
****
is the result of the matrix multiply.
If lhs and rhs are two dimensional, it has sizes (m,p).
Otherwise it has the sizes (\*,m,p) that result from broadcasting.

Example
*******
{xrst_literal ,
    examples/adten/matmul.cpp
    BEGIN_CPP, END_CPP
}
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

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` .
Note that the case were both arguments are one dimensional (i.e, vectors)
has been converted to an element-wise multiply and sum.

{xrst_end adten_matmul_dev}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR

// BEGIN_MATMUL BEGIN_DEV_MATMUL
adten_t adten_t::matmul(const adten_t& rhs) const
// END_MATMUL END_DEV_MATMUL
{
# ifndef NDEBUG
    c10::IntArrayRef lhs_shape = this->sizes();
    c10::IntArrayRef rhs_shape = rhs.sizes();
    //
    dev::user_assert( lhs_shape.size() >= 2,
        "matmul: left hand operand is not two or more dimensional"
    );
    dev::user_assert( rhs_shape.size() >= 2,
        "matmul: right hand operand is not two or more dimensional"
    );
    size_t n_left  = lhs_shape.size();
    size_t n_right = rhs_shape.size();
    dev::user_assert( lhs_shape[n_left - 1] == rhs_shape[n_right - 2],
        "matmul: the number of columns is left operand "
        "not equal number of rows in right operand"
    );
# endif
    //
    // res_tensor
    at::Tensor res_tensor = at::matmul(m_at_ten, rhs.m_at_ten);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "matmul: "
        "left AD tensor's tape is not tape that is recording"
    );
    dev::user_assert( rhs.m_tape_id == tape.m_tape_id , "matmul: "
        "right AD tensor's tape is not tape that is recording"
    );
    //
    // res_adtype
    adtype_t res_adtype = std::max( m_adtype, rhs.m_adtype );
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
            assert( res_adtype == adtype_t::variable  && "AD tensor being "
                "multiplied is not constant, parameter, or variable"
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
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( rhs.m_index );
        agraph->m_arg_type.push_back( rhs.m_adtype );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
