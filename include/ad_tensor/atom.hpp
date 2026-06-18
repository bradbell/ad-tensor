#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent atom usr}

Atomic Functions
################

{xrst_toc_table
    src/adten/call_atom.cpp
}

{xrst_end atom}
-------------------------------------------------------------------------------
{xrst_begin atom_dev dev}

Atomic Functions Developer Documentation
########################################

{xrst_toc_table
    src/adten/call_atom.cpp
}

{xrst_end atom_dev}
-------------------------------------------------------------------------------
{xrst_begin atom_callback usr}
{xrst_spell
    rng
    adten
}

The Atomic Callback Class atom_callback_t
#########################################

atom_callback
*************
We use atom_callback below for an atom_callback_t
object created using its default constructor.

call_info
*********
This value is passed through to the callback functions
from :ref:`call_atom@call_info` in the call to this atomic function.

rng_used
********
If rng_used is the empty vector, all of the range values are used.
Otherwise this vector has the same size as :ref:`call_atom@arange`
in the corresponding call this atomic function.
If rng_used[i] is true,
the return value with index i is used and must be calculated.
If it is false,
the return value with index i is not used and need not be calculated.

domain
******
is the value of the domain tensors for this atomic function call.

set
***

name
====
After the atom_callback_t constructor,
the name for the atomic function is empty.
It must set this with a call of the form
{xrst_literal ,
    BEGIN_SET_NAME, END_SET_NAME
}

Functions
=========
The following is a list of the callback functions:

.. csv-table::
    :header-rows: 1

    callback,       Required when this atomic is used with
    depend,         always
    forward,        always
    forward_der,    :ref:`adfn_forward_der-name` with TensorType at::Tensor
    reverse_der,    :ref:`adfn_reverse_der-name` with TensorType at::Tensor
    ad_forward,     :ref:`adfn_forward_par-name` with TensorType adten
    ad_forward_der, :ref:`adfn_forward_der-name` with TensorType adten
    ad_reverse_der, :ref:`adfn_reverse_der-name` with TensorType adten

Note that ad_forward is also required when :ref:`adfn_forward_var-name`
is used with TensorType adten.

After the atom_callback_t constructor, all the callback functions are null.
Each callback function that is used must be set by a call of the form
{xrst_code cpp}
    atom_callback.set_function(const function_t& function)
{xrst_code}
where function is the callback function; e.g.,
{xrst_code cpp}
    atom_callback.set_depend(const function_t& depend)
{xrst_code}

depend
******
{xrst_literal ,
    BEGIN_DEPEND, END_DEPEND
}
This returns a dependency :ref:`sparsity-name` pattern
for this atomic function.
If the atomic function range index i depends on the domain index j,
then (i,j) is in the sparsity pattern for this atomic function.

forward
*******
{xrst_literal ,
    BEGIN_FORWARD_T, END_FORWARD_T
}
This computes the atomic function range values
as a function of its domain values.

{xrst_end atom_callback}
-------------------------------------------------------------------------------
{xrst_begin atom_global usr}

Global Object That holds All Atomic Callbacks
#############################################

atom_global
***********
{xrst_literal ,
    BEGIN_SINGLETON, END_SINGLETON
}

atom_id
*******
{xrst_literal ,
    BEGIN_STORE, END_STORE
}

{xrst_end atom_global}
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor {  // BEGIN_AD_TENSOR_NAMESPACE
// --------------------------------------------------------------------------
// atom_callback_t
class atom_callback_t {
public:
    //
    // BEGIN_DEPEND
    typedef sparsity_t (*depend_t)(size_t call_info);
    // END_DEPEND
    //
    // BEGIN_FORWARD_T
    typedef vector<at::Tensor> (*forward_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const at::Tensor*>&  domain
    );
    // END_FORWARD_T
    //
    // ad_forward_t
    typedef vector<adten_t> (*ad_forward_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const adten_t*>&     domain
    );
    //
    // forward_der_t
    typedef vector<at::Tensor> (*forward_der_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const at::Tensor*>&  domain    ,
        const vector<const at::Tensor*>&  dom_der
    );
    //
    // ad_forward_der_t
    typedef vector<adten_t> (*ad_forward_der_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const adten_t*>&     domain    ,
        const vector<const adten_t*>&     dom_der
    );
    //
    // reverse_der_t
    typedef vector<at::Tensor> (*reverse_der_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const at::Tensor*>&  domain    ,
        const vector<at::Tensor>&         rng_der
    );
    //
    // ad_reverse_der_t
    typedef vector<adten_t> (*ad_reverse_der_t) (
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<const adten_t*>&     domain    ,
        const vector<adten_t>&            rng_der
    );

private:
    // m_name
    std::string m_name;
    //
    // m_depend
    depend_t   m_depend;
    //
    // m_forward, m_forward_der, m_reverse_der
    forward_t         m_forward;
    forward_der_t     m_forward_der;
    reverse_der_t     m_reverse_der;
    //
    // m_ad_forward, m_ad_forward_der, m_ad_reverse_der
    ad_forward_t         m_ad_forward;
    ad_forward_der_t     m_ad_forward_der;
    ad_reverse_der_t     m_ad_reverse_der;
public:
    //
    // BEGIN_CTOR
    atom_callback_t(void)
    : m_name()
    , m_depend(nullptr)
    , m_forward(nullptr)
    , m_forward_der(nullptr)
    , m_reverse_der(nullptr)
    , m_ad_forward(nullptr)
    , m_ad_forward_der(nullptr)
    , m_ad_reverse_der(nullptr)
    // END_CTOR
    { }
    // BEGIN_SET_NAME
    void set_name(const std::string& name);
    // END_SET_NAME
    //
    // Function Setters
    void set_depend(const depend_t&                    depend);
    void set_forward(const forward_t&                  forward);
    void set_forward_der(const forward_der_t&          forward_der);
    void set_reverse_der(const reverse_der_t&          reverse_der);
    void set_ad_forward(const ad_forward_t&            ad_forward);
    void set_ad_forward_der(const ad_forward_der_t&    ad_forward_der);
    void set_ad_reverse_der(const ad_reverse_der_t&    ad_reverse_der);
    //
    // Getters
    const std::string&      get_name(void) const;
    const depend_t&         get_depend(void) const;
    const forward_t&        get_forward(void) const;
    const forward_der_t&    get_forward_der(void) const;
    const reverse_der_t&    get_reverse_der(void) const;
    const ad_forward_t&     get_ad_forward(void) const;
    const ad_forward_der_t& get_ad_forward_der(void) const;
    const ad_reverse_der_t& get_ad_reverse_der(void) const;
};
// atom_global_t
class atom_global_t {
private:
    std::shared_mutex          m_rw_mutex;
    vector<atom_callback_t>    m_atom_callback_vec;
    //
    // default constructor
    atom_global_t(void)
    { }
public:
    //
    // automatic copy constructor
    atom_global_t(const atom_global_t&) = delete;
    //
    //  automatic assignment operator
    void operator=(const atom_global_t&) = delete;
    //
    // BEGIN_SINGLETON
    // atom_global = atom_global_t::singleton()
    static atom_global_t& singleton(void)
    {   // END_SINGLETON
        static atom_global_t m_atom_global;
        return m_atom_global;
    }
    //
    // BEGIN_STORE
    // atom_id = atom_global.store(atom_callback)
    size_t store(const atom_callback_t& atom_callback);
    // END_STORE
    //
    // get
    const atom_callback_t& get(size_t atom_id);
};

} // END_AD_TENSOR_NAMESPACE
