#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn usr}

The AD Function Class
#####################

Default Constructor
*******************
This creates an empty AD function; i.e.,
one that does not have any operations:
{xrst_literal ,
    BEGIN_ADFN_T, END_ADFN_T
}

is_empty
********
Returns true if there are no operations in this AD function:
{xrst_literal ,
    BEGIN_IS_EMPTY, END_IS_EMPTY
}

name
****
get_name returns the name chosen by :ref`stop_recording-name` :
{xrst_literal ,
    BEGIN_GET_NAME, END_GET_NAME
}

trace
*****
{xrst_literal ,
    BEGIN_GET_TRACE, END_GET_TRACE
    BEGIN_SET_TRACE, END_SET_TRACE
}
get_trace returns the previous value chosen by set_trace.
The default value, when trace has not been set, is false.

Other Public Members
********************
{xrst_toc_table
    src/adfn/forward_dep.cpp
    src/adfn/forward_par.cpp
    src/adfn/forward_var.cpp
    src/adfn/get_range.cpp
    src/adfn/forward_der.cpp
    src/adfn/reverse_der.cpp
}


{xrst_end adfn}
------------------------------------------------------------------------------
{xrst_begin adfn_dev dev}
{xrst_spell
    rng
}

The AD Function Class Private Members
#####################################
{xrst_literal ,
    // BEGIN_PRIVATE, // END_PRIVATE
}

m_par
*****
is the acyclic graph for the dependent parameters.

m_var
*****
is the acyclic graph for the dependent variables.

m_con
*****
is the vector of constant tensors.

m_rng_index
***********
is the constant, parameter, or variable index for each of the range tensors.

m_rng_ad_type
*************
is the ad_type for each of the range tensors.

m_rng_shapes
************
is the shape for each of the range tensors.

{xrst_end adfn_dev}
------------------------------------------------------------------------------
*/
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
#include <ad_tensor/options.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/call_op_depend.hpp>
//
namespace ad_tensor { class adten_t; }
//
// BEGIN_ADFN_T
namespace ad_tensor { class adfn_t
// END_ADFN_T
{
    friend class adten_t;
private:
// BEGIN_PRIVATE
    dev::agraph_t             m_par;
    dev::agraph_t             m_var;
    vector<at::Tensor>        m_con;
    vector<size_t>            m_rng_index;
    vector<ad_type_t>         m_rng_ad_type;
    vector< vector<int64_t> > m_rng_shapes;
    options_t                 m_options;
// END_PRIVATE
public:
    //
    // BEGIN_DEFAULT_CTOR
    adfn_t()
    // END_DEFAULT_CTOR
    : m_par()
    , m_var()
    , m_con()
    , m_rng_index()
    , m_rng_ad_type()
    , m_rng_shapes()
    , m_options()
    { }
    //
    // BEGIN_IS_EMPTY
    bool is_empty(void) const
    // END_IS_EMPTY
    { return
        m_par.is_empty() &&
        m_var.is_empty() &&
        m_con.empty() &&
        m_rng_index.empty() &&
        m_rng_ad_type.empty() &&
        m_rng_shapes.empty() &&
        m_options.get_name().empty()
        ;
    }
    //
    // BEGIN_GET_NAME
    const std::string& get_name(void) const
    {   // END_GET_NAME
        return m_options.get_name();
    }
    //
    // BEGIN_GET_TRACE
    bool get_trace(void) const
    {   // END_GET_TRACE
        return m_options.get_trace();
    }
    //
    // BEGIN_SET_TRACE
    void set_trace(bool trace)
    {   // END_SET_TRACE
        m_options.set_trace(trace);
    }
    //
    // forward_dep
    std::tuple<sparsity_t, sparsity_t> forward_dep(void) const;
    //
    // forward_par
    template <class TensorType>
    vector<TensorType> forward_par(const vector<TensorType>& dom_par) const;
    //
    // forward_var
    template <class TensorType>
    vector<TensorType> forward_var(
        const vector<TensorType>& all_par ,
        const vector<TensorType>& dom_var ,
        const options_t&          options
    ) const;
    //
    // get_range
    template <class TensorType>
    vector<TensorType> get_range(
        const vector<TensorType>& all_par ,
        const vector<TensorType>& all_var
    ) const;
    //
    // forward_der
    template <class TensorType>
    vector<TensorType> forward_der(
        const vector<TensorType>& all_par ,
        const vector<TensorType>& all_var ,
        const vector<TensorType>& dom_der ,
        const options_t&          options
    ) const;
    //
    // reverse_der
    template <class TensorType>
    vector<TensorType> reverse_der(
        const vector<TensorType>& all_par ,
        const vector<TensorType>& all_var ,
        const vector<TensorType>& rng_der ,
        const options_t&          options
    ) const;
}; }
