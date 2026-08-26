// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_src_gen usr}
{xrst_spell
    dir
    cpp
}

Generate C++ Source Code A AD Function
######################################

Under Construction
******************
The documentation below is just a plan.
Its implementation is under construction.

Syntax
******
{xrst_code cpp}
    adfn.src_gen(dir)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_SRC_GEN, END_SRC_GEN
}

adfn
****
We use adfn for this AD function object.

dir
***
is the directory where the C++ and binary output files are stored.

adfn_name.cpp
=============
The C++ output file name begins with the name of this adfn
(see :ref:`stop_recording-name`) and the ends with ``.cpp`` .
It defines the source code function defined below.

adfn_name.binary
================
The binary output file name begins with the name of this adfn;
and the ends with ``.binary`` .
It contains the constant at::Tensors used to evaluate the
source code function defined below.

Source Code Function
********************
Is C++ source code for the function range = adfn(dom_var, dom_par) where:
{xrst_code cpp}
    par_all = adfn.forward_par(dom_par);
    var_all = adfn.forward_var(dom_var, par_all);
    range   = adfn.get_range(var_all, par_all);
{xrst_code}
The vector range above can contain derivatives if adfn
was recording using AD Tensors and derivatives of another AD function.

Syntax
======
{xrst_code cpp}
   range = adfn_name_plugin(dom_par, dom_var )
{xrst_code}

adfn_name_plugin
================
This begins with the name of this adfn and the ends with ``_plugin`` .

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
    ad_tensor::vector<at::Tensor> range
{xrst_code}

{xrst_end adfn_src_gen}
------------------------------------------------------------------------------
*/
#include <string>
#include <filesystem>
#include <fstream>
#include <format>
#include <ad_tensor/adfn.hpp>
//
namespace { // BEGIN_EMPTY_NAMESPACE
// string, format
using std::string;
using std::format;
using ad_tensor::vector;
using ad_tensor::adtype_t;
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
    {   constexpr const char* fmt =
R"|(// ad_tensor::adfn::src_gen output
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/no_elements.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
ad_tensor::vector<at::Tensor> {}_plugin(
const ad_tensor::vector<at::Tensor>& dom_par ,
const ad_tensor::vector<at::Tensor>& dom_var )
{{   //
    // dev, string, Tensor, vector, has_elements
    namespace dev = ad_tensor::dev;
    using std::string;
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::has_elements;
)|";
        src += std::format(fmt, adfn_name);
    }
    //
    // ifndef NDEBUG
    src += indent + "//\n";
    src += "#ifndef NDEBUG\n";
    //
    src += indent + "{\n";
    //
    // msg
    {   constexpr const char* fmt =
R"|(        //
        // msg
        string msg = "{}_plugin: ";
        //
)|";
        src += std::format(fmt, adfn_name);
    }
    //
    // par_shapes
    src += indent + indent + "//  par_shapes\n";
    src += indent + indent + "vector< vector<int64_t> > par_shapes;\n";
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
    src +=
R"|(        if( dom_par.size() != par_shapes.size() ) {
            msg += "dom_par.size() = " + std::to_string( dom_par.size() );
            msg += " and its size for this adfn is ";
            msg += std::to_string( par_shapes.size() );
            dev::user_assert( false , msg );
        }
        for(size_t i = 0; i < par_shapes.size(); ++i) {
            c10::IntArrayRef shape = par_shapes[i];
            if( has_elements(dom_par[i]) ) {
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
    // var_shapes
    src += indent + indent + "//  var_shapes\n";
    src += indent + indent + "vector< vector<int64_t> > var_shapes;\n";
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
    src +=
R"|(        if( dom_var.size() != var_shapes.size() ) {
            msg += "dom_var.size() = " + std::to_string( dom_var.size() );
            msg += " and its size for this adfn is ";
            msg += std::to_string( var_shapes.size() );
            dev::user_assert( false , msg );
        }
        for(size_t i = 0; i < var_shapes.size(); ++i) {
            c10::IntArrayRef shape = var_shapes[i];
            if( has_elements(dom_var[i]) ) {
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
    // endif
    src += indent + "}\n";
    src += "#endif\n";
    //
    return src;
}
// -----------------------------------------------------------------------

} // END_EMPTY_NAMESPACE
// -------------------------------------------------------------------------
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
// BEGIN_SRC_GEN
void adfn_t::src_gen(const std::string& dir) const
{   // END_SRC_GEN
    //
    // fs
    namespace fs = std::filesystem;
    //
    // adfn_name
    string adfn_name = get_name();
    //
    // n_par_dom, n_var_dom
    size_t n_par_dom = m_par.m_dom_shapes.size();
    size_t n_var_dom = m_var.m_dom_shapes.size();
    //
    // dir_path
    fs::path dir_path(dir);
    //
    // file_cpp
    fs::path file_cpp_path = dir_path / ( adfn_name + ".cpp" );
    std::ofstream file_cpp(file_cpp_path);
    //
    // file_binary_path
    fs::path file_binary_path = dir_path / ( adfn_name + ".binary" );
    //
    // adfn_name.binary
    {   string file_name = file_binary_path.string();
        auto vec_ten_ptr =
            dynamic_cast< const std::vector<at::Tensor>* >(&m_con_vec);
        torch::save( *vec_ten_ptr, file_name);
    }
    //
    // file_cpp: dev, string, Tensor, vector, has_elements
    // also check size and shapes for dom_par and dom_var
    file_cpp <<  preamble(
        adfn_name          ,
        m_par.m_dom_shapes ,
        m_var.m_dom_shapes
    );
    //
    // file_cpp: adfn_name, no_elem
    {   constexpr const char* fmt =
R"|(    //
    // adfn_name, no_elem
    string adfn_name = "{}";
    Tensor no_elem = ad_tensor::no_elements();
)|";
        file_cpp << std::format(fmt, adfn_name);
    }
    //
    // file_cpp: con_vec
    {   string file_name = fs::canonical(file_binary_path).string();
        constexpr const char* fmt =
R"|(    //
    // con_vec
    vector<Tensor> con_vec;
    {{   string file_name = "{}";
        auto vec_ptr = dynamic_cast< std::vector<Tensor>* > (&con_vec);
        try {{
            torch::load(*vec_ptr, file_name);
        }} catch (...) {{
            dev::user_assert(false,
                adfn_name + "_plugin could not load " + file_name
            );
        }}
    }}
)|";
        file_cpp << std::format(fmt, file_name);
    }
    //
    // file.cpp: par_dep, var_dep
    {
        assert( n_par_dom <= m_par.m_op_seq.size() );
        assert( n_var_dom <= m_var.m_op_seq.size() );
        //
        size_t n_par_dep = m_par.m_op_seq.size() - n_par_dom;
        size_t n_var_dep = m_var.m_op_seq.size() - n_var_dom;
        //
        constexpr const char* fmt =
R"|(    //
    // par_dep, var_dep
    size_t n_par_dep = {};
    size_t n_var_dep = {};
    vector<Tensor> par_dep(n_par_dep, no_elem);
    vector<Tensor> var_dep(n_var_dep, no_elem);
)|";
        file_cpp << std::format(fmt, n_par_dep, n_var_dep);
    }
    //
    // tensor_src
    auto tensor_src = [n_par_dom, n_var_dom] (size_t index, adtype_t adtype) {
        string result;
        switch(adtype) {
            //
            case adtype_t::constant: {
                constexpr const char* fmt = "con_vec[{}]";\
                result = std::format(fmt, index);
            }
            //
            case adtype_t::parameter: {
                if( index < n_par_dom ) {
                    constexpr const char* fmt = "dom_par[{}]";\
                    result = std::format(fmt, index);
                } else {
                    constexpr const char* fmt = "par_dep[{}]";\
                    result = std::format(fmt, index - n_par_dom);
                }
            }
            //
            case adtype_t::variable: {
                if( index < n_var_dom ) {
                    constexpr const char* fmt = "dom_var[{}]";\
                    result = std::format(fmt, index);
                } else {
                    constexpr const char* fmt = "var_dep[{}]";\
                    result = std::format(fmt, index - n_var_dom);
                }
            }
            default:
            assert( false );
        }
    };
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // file_cpp: range
    {   constexpr const char* fmt =
R"|(    //
    // range
    size_t n_range = {};
    vector<Tensor> range(n_range, no_elem);
    //
    return range;
)|";
        file_cpp << std::format(fmt, m_rng_index.size());
    }
    //
    // file_cpp
    file_cpp << "}\n";
    file_cpp.close();
}
} // END_AD_TENSOR_NAMESPACE
