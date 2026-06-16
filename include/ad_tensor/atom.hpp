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
{xrst_begin atom_callback usr}

The Atomic Callback Class atom_callback_t
#########################################

atom_callback
*************
we use atom_callback below for an atom_callback_t
object created using its default constructor.

call_info
*********
the meaning of this is special for the original call
to this atomic function.

domain
******
is the value of the domain variables for this atomic function call.

set
***
After the atom_callback_t constructor, all the callback functions are null.
Each of the callback functions can be set by a call of the form
{xrst_code cpp}
    atom_callback.set_name(const name_t& name)
{xrst_code}
where name is the callback name; e.g., depend.

depend
******
{xrst_literal ,
    BEGIN_DEPEND, END_DEPEND
}
This callback is required for all atomic functions.
It returns a dependency :ref:`sparsity-name` pattern
for this atomic function.
If the atomic function range index i depends on the domain index j,
then (i,j) is in the sparsity pattern for this atomic function.

forward
*******
{xrst_literal ,
    BEGIN_FORWARD, END_FORWARD
}
This callback is required for all atomic functions.
It computes the range values for the atomic function.

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
    typedef sparsity_t (*depend_t)(int64_t call_info);
    // END_DEPEND
    //
    // BEGIN_FORWARD
    typedef vector<at::Tensor> (*forward_t) (
        int64_t            call_info ,
        vector<at::Tensor> domain
    );
    // END_FORWARD
    //
    // ad_forward_t
    typedef vector<adten_t> (*ad_forward_t) (
        int64_t            call_info ,
        vector<adten_t>    domain
    );
    //
    // forward_der_t, ad_forward_der_t
    typedef vector<at::Tensor> (*forward_der_t) (
        int64_t            call_info ,
        vector<at::Tensor> domain ,
        vector<at::Tensor> domain_der
    );
    typedef vector<adten_t> (*ad_forward_der_t) (
        int64_t            call_info ,
        vector<adten_t>    domain ,
        vector<adten_t>    domain_der
    );
    //
    // reverse_der_t, ad_reverse_der_t
    typedef vector<at::Tensor> (*reverse_der_t) (
        int64_t            call_info ,
        vector<at::Tensor> domain    ,
        vector<at::Tensor> range_der
        );
    typedef vector<adten_t> (*ad_reverse_der_t) (
        int64_t            call_info ,
        vector<adten_t>    domain    ,
        vector<adten_t>    range_der
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
    //
    // Setters
    void set_name(const std::string&                   name);
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
