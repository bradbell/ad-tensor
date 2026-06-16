#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
//
namespace ad_tensor {  // BEGIN_AD_TENSOR_NAMESPACE
// --------------------------------------------------------------------------
// atom_t
class atom_t {
public:
    //
    // depend_t
    typedef sparsity_t (*depend_t)(int64_t call_info);
    //
    // forward_t, ad_forward_t
    typedef vector<at::Tensor> (*forward_t) (
        int64_t            call_info ,
        vector<at::Tensor> domain
    );
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
    atom_t(void)
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
// atom_info_t
class atom_info_t {
private:
    std::shared_mutex m_rw_mutex;
    vector<atom_t>    m_atom_vec;
    //
    // default constructor
    atom_info_t(void)
    { }
public:
    //
    // automatic copy constructor
    atom_info_t(const atom_info_t&) = delete;
    //
    //  automatic assignment operator
    void operator=(const atom_info_t&) = delete;
    //
    // singleton
    static atom_info_t& singleton(void)
    {   static atom_info_t m_atom_info;
        return m_atom_info;
    }
    //
    // store
    size_t store(const atom_t& adfn);
    //
    // get
    const atom_t& get(size_t atom_id);
};

} // END_AD_TENSOR_NAMESPACE
