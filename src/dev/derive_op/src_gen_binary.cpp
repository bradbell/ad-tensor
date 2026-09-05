// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin src_gen_binary dev}
{xrst_spell
    op
    newline
}

Generate Source Code for Binary Operators
#########################################

Syntax
******
{xrst_code cpp}
    src = src_gen_binary(op_index, agraph, variable_agraph, tensor_src);
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_SRC_GEN_BINARY, END_SRC_GEN_BINARY
}

op_index
********
Pis the index of this operator in the operation sequence.
It is also the index of the result for this operator.
It must be greater than zero, because the first result is an
independent parameter tensor.

agraph
******
Is the :ref:`agraph-name` for the operation sequence that we are using.

variable_agraph
***************
if this is true (false) the acyclic graph is for variables (parameters).

tensor_src
**********
The function call
{xrst_code cpp}
    src = tensor_src(index, adtype)
{xrst_code}
returns a source code representation of a
constant, parameter of variable (depending on adtype) with the specified index.
The return, src, can be used to set or get the corresponding value.
There are no newline characters in src; hence it is one line of source code.

src
===
Is the source code corresponding to the binary operator at index op_index.

{xrst_end src_gen_binary}
*/
#include <cassert>
#include <format>
#include <string>
#include <functional>
#include <ad_tensor/adtype.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
namespace ad_tensor { namespace dev { // ad_tensor::dev
//
// BEGIN_SRC_GEN_BINARY
std::string src_gen_binary(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src      )
{   // END_SRC_GEN_BINARY
    //
    // string
    using std::string;
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 && "add: n_arg != 2" );
# endif
    //
    // lhs_src
    size_t   lhs_index   = agraph.m_arg_value[arg_start];
    adtype_t lhs_adtype  = agraph.m_arg_type[arg_start];
    string   lhs_src     = tensor_src(lhs_index, lhs_adtype);
    //
    // rhs_src
    size_t   rhs_index   = agraph.m_arg_value[arg_start + 1];
    adtype_t rhs_adtype  = agraph.m_arg_type[arg_start + 1];
    string   rhs_src     = tensor_src(rhs_index, rhs_adtype);
    //
    // target_src
    size_t   target_index  = op_index;
    adtype_t target_adtype =
        variable_agraph ? adtype_t::variable : adtype_t::parameter;
    string   target_src    = tensor_src(target_index, target_adtype);
    //
    // op_enum
    op_enum_t op_enum = agraph.m_op_seq[op_index];
    //
    // src
    string src;
    switch( op_enum ) {
        //
        case op_enum_t::add: {
            constexpr const char* fmt = "{} = {} + {};";
            src = std::format(fmt, target_src, lhs_src, rhs_src);
        }
        break;
        //
        case op_enum_t::sub: {
            constexpr const char* fmt = "{} = {} - {};";
            src = std::format(fmt, target_src, lhs_src, rhs_src);
        }
        break;
        //
        case op_enum_t::mul: {
            constexpr const char* fmt = "{} = {} * {};";
            src = std::format(fmt, target_src, lhs_src, rhs_src);
        }
        break;
        //
        case op_enum_t::div: {
            constexpr const char* fmt = "{} = {} / {};";
            src = std::format(fmt, target_src, lhs_src, rhs_src);
        }
        break;
        //
        case op_enum_t::matmul: {
            constexpr const char* fmt = "{} = {}.matmul( {} );";
            src = std::format(fmt, target_src, lhs_src, rhs_src);
        }
        break;
        //
        default: {
            std::cerr << "src_gen_binary: the " << to_string(op_enum);
            std::cerr << " is not implemented\n";
            assert(false);
        }
    }
    return src;
}
} } // ad_tensor::dev
