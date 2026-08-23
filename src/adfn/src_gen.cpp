// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_src_gen usr}
{xrst_spell
}

Generate C++ Source Code For this AD Function
#############################################

Under Construction
******************
The documentation below is just a plan.
Its implementation is under construction.

Syntax
******
{xrst_code cpp}
    source_code = adfn.src_gen()
{xrst_code}

adfn
****
We use adfn for this AD function object.

source_code
***********
Is C++ source code for the function range = adfn(dom_var, dom_par) where:
{xrst_code cpp}
    par_all = adfn.forward_par(dom_par);
    var_all = adfn.forward_var(dom_var, par_all);
    y       = adfn.get_range(var_all, par_all);
{xrst_code}
The vector y above can contain derivatives if adfn
was recording using AD Tensors and derivatives of another AD function.

Syntax
======
{xrst_code cpp}
   range = adfn_name_plugin(dom_par, dom_var )
{xrst_code}

adfn_name_plugin
================
This begins with the name of this adfn; see :ref:`stop_recording-name`
and the ends with ``_plugin`` .

dom_par
=======
is the vector of domain parameter tensors and has the following prototype:
{xrst_code cpp}
    const ad_tensor::vector<at::Tensor>& dom_par;
{xrst_code}

dom_var
=======
is the vector of domain variable tensors and has the following prototype:
{xrst_code cpp}
    const ad_tensor::vector<at::Tensor>& dom_var;
{xrst_code}

range
=====
is the vector of range tensors and has the following prototype:
{xrst_code cpp}
    ad::tensor::vector<at::Tensor> range
{xrst_code}

{xrst_end adfn_src_gen}
------------------------------------------------------------------------------
*/
#include <string>
#include <format>
#include <ad_tensor/adfn.hpp>
//
namespace { // BEGIN_EMPTY_NAMESPACE
// string, format
using std::string;
using std::format;
using ad_tensor::vector;
//
// -----------------------------------------------------------------------
// preamble
std::string preamble(
    const std::string&               adfn_name ,
    const vector< vector<int64_t> >& par_shapes,
    const vector< vector<int64_t> >& var_shapes)  {
    //
    // indent
    string indent = "    ";
    //
    // src
    string src;
    //
    // src
    // Note that {{ and }} escapes the special meaning of {} in format.
    constexpr const char* fmt1 =
R"|(// ad_tensor::adfn::src_gen output
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
ad::tensor::vector<at::Tensor> {}_plugin(
const ad::tensor::vector<at::Tensor>& dom_par ,
const ad::tensor::vector<at::Tensor>& dom_var )
{{   //
)|";
    src += std::format(fmt1, adfn_name);
    //
    src += indent + "//\n";
    src += "#ifndef NDEBUG\n";
    //
    src += indent + "{\n";
    src += indent + indent + "//  par_shapes\n";
    src += indent + indent + "const vector< vector<int64_t> > par_shapes;\n";
    for(size_t i = 0; i < par_shapes.size(); ++i) {
        src +=  indent + indent + "par_shapes.push_back( {";
        for(size_t j = 0; j < par_shapes[i].size(); ++j) {
            src += std::to_string( par_shapes[i][j] );
            if( j+1 < par_shapes[i].size() ) {
                src += ", ";
            }
        }
        src += "} );\n";
    }
    //
    src += indent + indent + "//  var_shapes\n";
    src += indent + indent + "const vector< vector<int64_t> > var_shapes;\n";
    for(size_t i = 0; i < var_shapes.size(); ++i) {
        src +=  indent + indent + "var_shapes.push_back( {";
        for(size_t j = 0; j < var_shapes[i].size(); ++j) {
            src += std::to_string( var_shapes[i][j] );
            if( j+1 < var_shapes[i].size() ) {
                src += ", ";
            }
        }
        src += "} );\n";
    }
    // msg
    constexpr const char* fmt2 =
R"|(        //
        // msg
        string msg = "{}_plugin: ";
        //
)|";
    src += std::format(fmt2, adfn_name);
    //
    // check par_shapes
    src +=
R"|(        if( dom_par.size() != par_shapes.size() ) {
            msg += "dom_par.size() = " + std::to_string( dom_par.size() );
            msg += " and its size for this adfn is ";
            msg += std::to_string( par_shapes.size() );
            dev::user_assert( false , msg );
        }
        for(size_t i = 0; i < par_shapes.size(); ++i) {
            c10::IntArrayRef shape = par_shapes[i];
            if( dom_par[i].defined() ) {
                if( ! dom_par[i].sizes().equals( shape ) ) {
                    msg += "dom_par[" + std::to_string(i) + "] shape is ";
                    msg += dev::to_string( dom_par[i].sizes() );
                    msg += " and its shape for this adfn is ";
                    msg += dev::to_string( shape );
                    dev::user_assert( false , msg );
                }
            }
        }
)|";
    //
    // check var_shapes
    src +=
R"|(        if( dom_var.size() != var_shapes.size() ) {
            msg += "dom_var.size() = " + std::to_string( dom_var.size() );
            msg += " and its size for this adfn is ";
            msg += std::to_string( var_shapes.size() );
            dev::user_assert( false , msg );
        }
        for(size_t i = 0; i < var_shapes.size(); ++i) {
            c10::IntArrayRef shape = var_shapes[i];
            if( dom_var[i].defined() ) {
                if( ! dom_var[i].sizes().equals( shape ) ) {
                    msg += "dom_var[" + std::to_string(i) + "] shape is ";
                    msg += dev::to_string( dom_var[i].sizes() );
                    msg += " and its shape for this adfn is ";
                    msg += dev::to_string( shape );
                    dev::user_assert( false , msg );
                }
            }
        }
)|";
    //
    src += indent + "}\n";
    src += "#endif\n";
    //
    return src;
}
// -----------------------------------------------------------------------

} // END_EMPTY_NAMESPACE
// -------------------------------------------------------------------------
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
//
std::string adfn_t::src_gen(void) const {
    //
    // indent
    string indent = "    ";
    //
    // src
    string src = preamble(
        get_name()         ,
        m_par.m_dom_shapes ,
        m_var.m_dom_shapes
    );
    //
    src += "}\n";
    return src;
}
} // END_AD_TENSOR_NAMESPACE
