// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_conv1d usr}
{xrst_spell
}

One Dimensional Cross Correlation
#################################

Syntax
******
{xrst_code cpp}
    output = conv1d(input, weight, bias, options)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_CONV1D_AD_TEN, END_CONV1D_AD_TEN
}

{xrst_end adten_conv1d}
-------------------------------------------------------------------------------
{xrst_begin adten_conv1d dev}
{xrst_spell
    rhs
}

Compute and Record Solution of Linear Equations
###############################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_CONV1D, END_CONV1D
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

    arg_index, arg_value,                  arg_type
    start + 0, index for input,            type for input
    start + 1, index for weight,           type for weight
    start + 3, index for bias,             type for bias
    start + 4, stride                      none
    start + 5, dilation                    none
    start + 6, groups                      none

where start be the length of arg_value and arg_type before this call to
``adten_t::binary`` .

{xrst_end adten_conv1d}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/no_elements.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // Begin ad_tensor::dev

adten_t adten_t::conv1d(
    const adten_t&                                  weight  ,
    const adten_t&                                  bias    ,
    const torch::nn::functional::Conv1dFuncOptions& options ) const
{
    //
# ifndef NDEBUG
    const adten_t&   input       = *this;
    c10::IntArrayRef input_shape = input.sizes();
    dev::user_assert( input_shape.size() == 3 ,
        "conv1d: input is not three dimensional"
    );
    //
    c10::IntArrayRef weight_shape = weight.sizes();
    dev::user_assert( weight_shape.size() == 3 ,
        "conv1d: weight is not three dimensional"
    );
    dev::user_assert( weight_shape[1] == input_shape[1],
        "conv1d: number input channels in input and weight is different"
    );
    //
# endif
    // res_tensor
    at::Tensor res_tensor;
    if( no_elements( bias ) ) {
        res_tensor = torch::nn::functional::conv1d(
            input.at_ten(), weight.at_ten(), options
        );
    } else {
        res_tensor = ad_tensor::conv1d(
            input.at_ten(), weight.at_ten(), bias.at_ten(), options
        );
    }
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "conv1d: "
        "input AD tensor's tape is not tape that is recording"
    );
    dev::user_assert( weight.m_tape_id == tape.m_tape_id , "conv1d: "
        "rhs AD tensor's tape is not tape that is recording"
    );
    dev::user_assert( bias.m_tape_id == tape.m_tape_id , "conv1d: "
        "bias AD tensor's tape is not tape that is recording"
    );
    //
    // res_adtype
    adtype_t res_adtype = std::max( m_adtype, weight.m_adtype );
    res_adtype          = std::max( res_adtype, bias.m_adtype );
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
            assert( res_adtype == adtype_t::variable  && "AD tensor in "
                "conv1d is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::conv1d );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( weight.m_index );
        agraph->m_arg_type.push_back( weight.m_adtype );
        //
        agraph->m_arg_value.push_back( bias.m_index );
        agraph->m_arg_type.push_back( bias.m_adtype );
        //
        agraph->m_arg_value.push_back( size_t( (*options.stride())[0] ) );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( size_t( (*options.dilation())[0] ) );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( size_t( options.groups() ) );
        agraph->m_arg_type.push_back( adtype_t::none );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
adten_t conv1d(
    const adten_t&                                  input   ,
    const adten_t&                                  weight  ,
    const adten_t&                                  bias    ,
    const torch::nn::functional::Conv1dFuncOptions& options )
{   dev::user_assert( ! options.bias().defined(),
        "cov21d: bias is and explicit argument and is defined in options."
    );
    return input.conv1d(weight, bias, options);
}
at::Tensor conv1d(
    const at::Tensor&                               input   ,
    const at::Tensor&                               weight  ,
    const at::Tensor&                               bias    ,
    const torch::nn::functional::Conv1dFuncOptions& options )
{   dev::user_assert( ! options.bias().defined(),
        "cov21d: bias is and explicit argument and is defined in options."
    );
    torch::nn::functional::Conv1dFuncOptions copy_options = options;
    copy_options.bias( bias );
    return torch::nn::functional::conv1d(
        input, weight, copy_options
    );
}

// ---------------------------------------------------------------------------
} // End ad_tensor
