// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_pad usr}
{xrst_spell
}

Padding a Tensor
################

Syntax
******
{xrst_code cpp}
    output = pad(input, options)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_PAD, END_PAD
}

Example
*******
{xrst_literal ,
    examples/adten/pad.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end adten_pad}
-------------------------------------------------------------------------------
{xrst_begin adten_pad_dev dev}
{xrst_spell
    enum
    op
}

Compute and Record Tensor Padding
#################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_PAD, END_PAD
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

    arg_index, arg_value,                   arg_type
    start + 0, index for input,             type for input
    start + 1, index for constant_value,    constant
    start + 2, mode_size_t,                 none
    start + 3, n_sizes,                     none
    start + 4, first padding size,          none
    ..., ..., ...
    start + n_sizes + 3, last padding size, none
    ...

where start be the length of arg_value and arg_type before this call to
``adten_t::binary`` .

Note that mode_size_t can be converted to an pad_enum_t using
{xrst_code cpp}
    pad_enum_t mode = static_cast<op_enum_t>(mode_size_t)
{xrst_code}

{xrst_end adten_pad_dev}
STOPPED HERE
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/no_elements.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/dev/pad_enum.hpp>

namespace ad_tensor { // Begin ad_tensor::dev

adten_t adten_t::pad(
    const torch::nn::functional::PadFuncOptions& options ) const {
    //
    // input
    const adten_t&   input       = *this;
    //
    // pad_sizes
    c10::IntArrayRef pad_sizes = options.pad();
    //
# ifndef NDEBUG
    c10::IntArrayRef input_shape = input.sizes();
    dev::user_assert( pad_sizes.size() <= 6 ,
        "pad: sizes in options is greater than 6"
    );
    dev::user_assert( pad_sizes.size() % 2 == 0 ,
        "pad: number of sizes is not even"
    );
    dev::user_assert( pad_sizes.size() / 2 <= input_shape.size() ,
        "pad: number of sizes divided by 2 > number of dimesiosn in input"
    );
# endif
    //
    // res_tensor
    at::Tensor res_tensor =
        torch::nn::functional::pad(input.at_ten(), options);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording ) {
        return adten_t( res_tensor );
    }
    dev::user_assert( input.m_tape_id == tape.m_tape_id , "pad: "
        "input AD tensor's tape is not tape that is recording"
    );
    //
    // res_adtype
    adtype_t res_adtype =  input.m_adtype;
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
        // mode_size_t, constant_index
        auto mode             = options.mode();
        size_t constant_index = 0;
        assert( no_elements( tape.m_con_vec[constant_index] ) );
        dev::pad_enum_t pad_enum = dev::pad_enum_t::constant;
        if( std::holds_alternative<torch::enumtype::kConstant>(mode) ) {
            pad_enum = dev::pad_enum_t::constant;
            double     value          = options.value();
            at::Tensor constant_value = torch::tensor(
                value , torch::TensorOptions().dtype(torch::kFloat64)
            );
            assert( constant_value.numel() == 1 );
            constant_index = tape.m_con_vec.size();
            tape.m_con_vec.push_back( std::move(constant_value) );
        }
        else if( std::holds_alternative<torch::enumtype::kReflect>(mode) )
            pad_enum = dev::pad_enum_t::reflect;
        else if( std::holds_alternative<torch::enumtype::kReplicate>(mode) )
            pad_enum = dev::pad_enum_t::replicate;
        else if( std::holds_alternative<torch::enumtype::kCircular>(mode) )
            pad_enum = dev::pad_enum_t::circular;
        else {
            assert(false && "pad: invalid result for options.mode()" );
        }
        size_t mode_size_t = static_cast<size_t>(pad_enum);
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "AD tensor in "
                "pad is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index
        res_index       = agraph->m_op_seq.size();
        //
        // agraph
        agraph->m_op_seq.push_back( dev::op_enum_t::pad );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( input.m_index );
        agraph->m_arg_type.push_back( input.m_adtype );
        //
        agraph->m_arg_value.push_back( constant_index );
        agraph->m_arg_type.push_back( adtype_t::constant );
        //
        agraph->m_arg_value.push_back( mode_size_t );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( pad_sizes.size() );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        for(size_t i = 0; i < pad_sizes.size(); ++i)
        {   agraph->m_arg_value.push_back( pad_sizes[i] );
            agraph->m_arg_type.push_back( adtype_t::none );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
adten_t pad(
    const adten_t&                               input   ,
    const torch::nn::functional::PadFuncOptions& options )  {
    return input.pad(options);
}

// ---------------------------------------------------------------------------
} // End ad_tensor
