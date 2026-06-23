// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
#define UNPACK \
    size_t arg_start = agraph.m_arg_start[op_index];\
    size_t atom_id   = agraph.m_arg_value[arg_start + 0]; \
    size_t call_info = agraph.m_arg_value[arg_start + 1]; \
    size_t n_domain  = agraph.m_arg_value[arg_start + 2]; \
    size_t n_range   = agraph.m_arg_value[arg_start + 3]; \
    size_t n_result  = agraph.m_arg_value[arg_start + 4]; \

//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
// ------------------------------------------------------------------------
// forward_par
template<> void call_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_vec
) const {
    //
    // rng_used, domain
    thread_local vector<bool>       rng_used;
    thread_local vector<at::Tensor> domain;
    //
    // forward_t
    typedef atom_callback_t::forward_t forward_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // options, forward
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options       = atom_callback.get_options();
    const forward_t&       forward       = atom_callback.get_forward();
    //
    // domain
    domain.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t arg_index = arg_start + 5 + j;
        domain.push_back( tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec
        ) );
    };
    //
    // rng_used
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
    }
    //
    // range
    vector<at::Tensor> range = forward(options, call_info, rng_used, domain);
    //
    // par_vec
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index      = arg_start + 5 + n_domain + k;
        size_t rng_index      = agraph.m_arg_value[arg_index];
        par_vec[op_index + k] = range[ rng_index ];
    }
}
template<> void call_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_vec
) const {
    assert(false && "call_op: ad_forward_par not implemented");
}
// ------------------------------------------------------------------------
// forward_var
template<> void call_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_vec     ,
    vector<at::Tensor>&          var_vec
) const {
    //
    // rng_used, domain
    thread_local vector<bool>       rng_used;
    thread_local vector<at::Tensor> domain;
    //
    // forward_t
    typedef atom_callback_t::forward_t forward_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // forward
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options   = atom_callback.get_options();
    const forward_t&       forward       = atom_callback.get_forward();
    //
    // domain
    domain.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t arg_index = arg_start + 5 + j;
        domain.push_back( tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        ) );
    };
    //
    // rng_used
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
    }
    //
    // range
    vector<at::Tensor> range = forward(options, call_info, rng_used, domain);
    //
    // par_vec
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index      = arg_start + 5 + n_domain + k;
        size_t rng_index      = agraph.m_arg_value[arg_index];
        var_vec[op_index + k] = range[ rng_index ];
    }
}
template<> void call_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_vec     ,
    vector<adten_t>&             var_vec
) const {
    //
    // domain
    thread_local vector<adten_t>    domain;
    //
    // arg_start. atom_id, call_info, n_domain, n_result
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t atom_id   = agraph.m_arg_value[arg_start + 0];
    size_t call_info = agraph.m_arg_value[arg_start + 1];
    size_t n_domain  = agraph.m_arg_value[arg_start + 2];
    size_t n_result  = agraph.m_arg_value[arg_start + 4];
    //
    // domain
    domain.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t arg_index = arg_start + 5 + j;
        domain.push_back( tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        ) );
    };
    //
    // range
    vector<adten_t>  range = adten_t::call_atom(atom_id, call_info, domain);
    //
    // par_vec
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index      = arg_start + 5 + n_domain + k;
        size_t rng_index      = agraph.m_arg_value[arg_index];
        var_vec[op_index + k] = range[ rng_index ];
    }
}
// ------------------------------------------------------------------------
// forward_der
template<> void call_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_vec     ,
    const vector<at::Tensor>&    var_vec     ,
    vector<at::Tensor>&          for_der
) const {
    //
    // rng_used, domain, dom_der
    thread_local vector<bool>       rng_used;
    thread_local vector<at::Tensor> domain;
    thread_local vector<at::Tensor> dom_der;
    //
    // forward_der_t
    typedef atom_callback_t::forward_der_t forward_der_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // forward_der
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options   = atom_callback.get_options();
    const forward_der_t&   forward_der   = atom_callback.get_forward_der();
    //
    // rng_used
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
    }
    //
    // domain, dom_der
    domain.resize(0);
    dom_der.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index = arg_start + 5 + j;
        size_t    vec_index = agraph.m_arg_value[arg_index];
        ad_type_t arg_type  = agraph.m_arg_type[arg_index];
        switch( arg_type ) {
            case ad_type_t::constant: {
                const at::Tensor& domain_j = con_vec[vec_index];
                domain.push_back( domain_j );
                dom_der.push_back( torch::zeros( domain_j.sizes() ) );
            }
            break;
            case ad_type_t::parameter: {
                const at::Tensor& domain_j = par_vec[vec_index];
                domain.push_back( domain_j );
                dom_der.push_back( torch::zeros( domain_j.sizes() ) );
            }
            break;
            case ad_type_t::variable: {
                const at::Tensor& domain_j = var_vec[vec_index];
                domain.push_back( domain_j );
                dom_der.push_back( for_der[vec_index] );
            }
            break;
            //
            default:
            assert(false);
            break;
        }
    }
    //
    // rng_der
    vector<at::Tensor> rng_der = forward_der(
        options, call_info, rng_used, domain, dom_der
    );
    //
    // for_der
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index      = arg_start + 5 + n_domain + k;
        size_t rng_index      = agraph.m_arg_value[arg_index];
        for_der[op_index + k] = rng_der[ rng_index ];
    }
}
template<> void call_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_vec     ,
    const vector<adten_t>&       var_vec     ,
    vector<adten_t>&             for_der
) const {
    //
    // rng_used, domain, dom_der
    thread_local vector<bool>    rng_used;
    thread_local vector<adten_t> domain;
    thread_local vector<adten_t> dom_der;
    //
    // forward_der_t
    typedef atom_callback_t::ad_forward_der_t forward_der_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // forward_der
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options   = atom_callback.get_options();
    const forward_der_t&   forward_der   = atom_callback.get_ad_forward_der();
    //
    // rng_used
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
    }
    //
    // domain, dom_der
    domain.resize(0);
    dom_der.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index = arg_start + 5 + j;
        size_t    vec_index = agraph.m_arg_value[arg_index];
        ad_type_t arg_type  = agraph.m_arg_type[arg_index];
        switch( arg_type ) {
            case ad_type_t::constant: {
                const at::Tensor& domain_j = con_vec[vec_index];
                c10::IntArrayRef shape = domain_j.sizes();
                domain.push_back( adten_t( domain_j ) );
                dom_der.push_back( adten_t( torch::zeros(shape) ) );
            }
            break;
            case ad_type_t::parameter: {
                const adten_t& domain_j = par_vec[vec_index];
                c10::IntArrayRef shape = domain_j.sizes();
                domain.push_back( domain_j );
                dom_der.push_back( adten_t(torch::zeros(shape) ) );
            }
            break;
            case ad_type_t::variable: {
                const adten_t& domain_j = var_vec[vec_index];
                domain.push_back( domain_j );
                dom_der.push_back( for_der[vec_index] );
            }
            break;
            //
            default:
            assert(false);
            break;
        }
    }
    //
    // rng_der
    vector<adten_t> rng_der = forward_der(
        options, call_info, rng_used, domain, dom_der
    );
    //
    // for_der
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index      = arg_start + 5 + n_domain + k;
        size_t rng_index      = agraph.m_arg_value[arg_index];
        for_der[op_index + k] = rng_der[ rng_index ];
    }
}
// ------------------------------------------------------------------------
// reverse_der
template<> void call_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_vec     ,
    const vector<at::Tensor>&    var_vec     ,
    vector<at::Tensor>&          rev_der
) const {
    //
    // rng_used, domain, rng_der
    thread_local vector<bool>       rng_used;
    thread_local vector<at::Tensor> domain;
    thread_local vector<at::Tensor> rng_der;
    //
    // reverse_der_t
    typedef atom_callback_t::reverse_der_t reverse_der_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // reverse_der
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options   = atom_callback.get_options();
    const reverse_der_t&   reverse_der   = atom_callback.get_reverse_der();
    //
    // rng_used, rng_der
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    rng_der.resize(0);
    rng_der.resize( n_range, torch::empty( {0} ) );
    for(size_t k = 0; k < n_result; ++k) {
        size_t rev_index    = op_index + k;
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
        rng_der[rng_index]  = rev_der[rev_index];
    }
    //
    // domain
    domain.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t arg_index = arg_start + 5 + j;
        domain.push_back( tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        ) );
    };
    //
    // dom_der
    vector<at::Tensor> dom_der = reverse_der(
        options, call_info, rng_used, domain, rng_der
    );
    //
    // rev_der
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index  = arg_start + 5 + j;
        ad_type_t ad_type    = agraph.m_arg_type[arg_index];
        if( ad_type == ad_type_t::variable ) {
            size_t rev_index = agraph.m_arg_value[arg_index];
            plus_equal( rev_der[rev_index], dom_der[j] );
        }
    }
}
template<> void call_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_vec     ,
    const vector<adten_t>&       var_vec     ,
    vector<adten_t>&             rev_der
) const {
    //
    // rng_used, domain, rng_der
    thread_local vector<bool>    rng_used;
    thread_local vector<adten_t> domain;
    thread_local vector<adten_t> rng_der;
    //
    // reverse_der_t
    typedef atom_callback_t::ad_reverse_der_t reverse_der_t;
    //
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    UNPACK
    //
    // reverse_der
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const atom_callback_t& atom_callback = atom_global.get( atom_id );
    const options_t&       options   = atom_callback.get_options();
    const reverse_der_t&   reverse_der   = atom_callback.get_ad_reverse_der();
    //
    // rng_used, rng_der
    rng_der.resize(0);
    rng_used.resize(0);
    rng_used.resize(n_range, false);
    rng_der.resize( n_range, adten_t( torch::empty( {0} ) ) );
    for(size_t k = 0; k < n_result; ++k) {
        size_t rev_index    = op_index + k;
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        rng_used[rng_index] = true;
        rng_der[rng_index]  = rev_der[rev_index];
    }
    //
    // domain
    domain.resize(0);
    for(size_t j = 0; j < n_domain; ++j) {
        size_t arg_index = arg_start + 5 + j;
        domain.push_back( tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        ) );
    };
    //
    // dom_der
    vector<adten_t> dom_der = reverse_der(
        options, call_info, rng_used, domain, rng_der
    );
    //
    // rev_der
    for(size_t j = 0; j < n_domain; ++j) {
        size_t    arg_index  = arg_start + 5 + j;
        ad_type_t ad_type    = agraph.m_arg_type[arg_index];
        if( ad_type == ad_type_t::variable ) {
            size_t rev_index = agraph.m_arg_value[arg_index];
            plus_equal( rev_der[rev_index], dom_der[j] );
        }
    }
}
} } // END_AD_TENSOR_DEV_NAMESPACE
