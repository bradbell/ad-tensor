#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <array>
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
#include <ad_tensor/direction.hpp>
#include <ad_tensor/adtype.hpp>
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
    vector<at::Tensor>        m_con_vec;
    dev::agraph_t             m_par;
    dev::agraph_t             m_var;
    vector<size_t>            m_rng_index;
    vector<adtype_t>          m_rng_adtype;
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
    //
    // optimize_con
    void optimize_con(
        const vector<bool>& depend_con ,
        const vector<bool>& depend_par ,
        const vector<bool>& depend_var
    );
public:
    // optimize
    void optimize(void);
    //
    // src_gen
    void src_gen(const std::string& plugin_dir) const;
    //
    // ctor
    adfn_t(void);
    //
    // is_empty
    bool is_empty(void) const;
    //
    // n_range
    size_t n_range(void) const;
    //
    // n_con
    size_t n_con(void) const;
    //
    // n_par
    size_t n_par(void) const;
    //
    // n_var
    size_t n_var(void) const;
    //
    // get_name
    const std::string& get_name(void) const;
    //
    // get_trace
    bool get_trace(void) const;
    //
    // set_trace
    void set_trace(bool trace);
    //
    // print_con
    void print_con(void) const;
    //
    // BEGIN_FORWARD_DEP
    std::array<sparsity_t, 2> forward_dep(void) const;
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
