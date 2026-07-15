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

print_con
*********
{xrst_literal ,
    BEGIN_PRINT_CON, END_PRINT_CON
}
print the constants stored in an AD function object.
This may be useful for understanding the trace output.

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
    BEGIN_MEMBER_DATA, END_MEMBER_DATA
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
#include <array>
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
#include <ad_tensor/direction.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/call_op_depend.hpp>
//
// adten_t, adfn_t
namespace ad_tensor { class adten_t; class adfn_t; }
//
// dev::rng_depend
// cannot use include/dev/optimize.hpp because it includes this file
namespace ad_tensor { namespace dev {
    std::array< vector<bool>, 3> rng_depend(
        const adfn_t* adfn
    );
} }
//
// BEGIN_ADFN_T
namespace ad_tensor { class adfn_t
// END_ADFN_T
{
    friend class adten_t;
    // ad_tensor::make_chkpnt
    // is a link to the private function make_chkpnt defined in this class
    friend size_t make_chkpnt(
        adfn_t&                      adfn,
        const vector<at::Tensor>&    domain,
        c10::ArrayRef<direction_t>   directions
    );
    // ad_tensor::rng_depend
    // is a link to the private function rng_depend defined in this class
    friend std::array< vector<bool>, 3 > dev::rng_depend(const adfn_t* adfn);
    //
private:
// BEGIN_MEMBER_DATA
    vector<at::Tensor>        m_con;
    dev::agraph_t             m_par;
    dev::agraph_t             m_var;
    vector<size_t>            m_rng_index;
    vector<ad_type_t>         m_rng_ad_type;
    vector< vector<int64_t> > m_rng_shapes;
    std::string               m_name;
    bool                      m_trace;
// END_MEMBER_DATA
    //
    static size_t make_chkpnt(
        adfn_t&                      adfn,
        const vector<at::Tensor>&    domain,
        c10::ArrayRef<direction_t>   directions
    );
    //
    std::array< vector<bool>, 3 > rng_depend
    (void) const;
public:
    //
    // BEGIN_DEFAULT_CTOR
    adfn_t()
    // END_DEFAULT_CTOR
    : m_con()
    , m_par()
    , m_var()
    , m_rng_index()
    , m_rng_ad_type()
    , m_rng_shapes()
    , m_name()
    , m_trace(false)
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
        m_name.empty()
        ;
    }
    //
    // BEGIN_GET_NAME
    const std::string& get_name(void) const
    {   // END_GET_NAME
        return m_name;
    }
    //
    // BEGIN_GET_TRACE
    bool get_trace(void) const
    {   // END_GET_TRACE
        return m_trace;
    }
    //
    // BEGIN_SET_TRACE
    void set_trace(bool trace)
    {   // END_SET_TRACE
        m_trace = trace;
    }
    //
    // BEGIN_PRINT_CON
    void print_con(void) const;
    // END_PRINT_CON
    //
    // BEGIN_FORWARD_DEP
    std::tuple<sparsity_t, sparsity_t> forward_dep(void) const;
    // END_FORWARD_DEP
    //
    // BEGIN_FORWARD_PAR
    template <class TensorType>
    vector<TensorType> forward_par(const vector<TensorType>& dom_par) const;
    // END_FORWARD_PAR
    //
    // BEGIN_FORWARD_VAR
    template <class TensorType>
    vector<TensorType> forward_var(
        const vector<TensorType>& dom_var ,
        const vector<TensorType>& all_par = vector<TensorType>()
    ) const;
    // END_FORWARD_VAR
    //
    // BEGIN_GET_RANGE
    template <class TensorType>
    vector<TensorType> get_range(
        const vector<TensorType>& all_var ,
        const vector<TensorType>& all_par = vector<TensorType>()
    ) const;
    // END_GET_RANGE
    //
    // BEGIN_FORWARD_DER
    template <class TensorType>
    vector<TensorType> forward_der(
        const vector<TensorType>& dom_der ,
        const vector<TensorType>& all_var ,
        const vector<TensorType>& all_par = vector<TensorType>()
    ) const;
    // END_FORWARD_DER
    //
    // BEGIN_REVERSE_DER
    template <class TensorType>
    vector<TensorType> reverse_der(
        const vector<TensorType>& rng_der ,
        const vector<TensorType>& all_var ,
        const vector<TensorType>& all_par = vector<TensorType>()
    ) const;
    // END_REVERSE_DER
}; }
