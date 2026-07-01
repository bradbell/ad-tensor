// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin call_op_depend dev}
{xrst_spell
    agraph
}

Compute Call Operator Dependencies
##################################

Prototype
*********
{xrst_literal ,
    BEGIN_CALL_OP_DEPEND, END_CALL_OP_DEPEND
}

n_set
*****
We use the notation n_set for the input value of vec_sets.n_set().

op_index
********
The index in agraph for this operator.
This must correspond to a call operator.

agraph
******
If n_set < n_par this is the acyclic operator graph for parameters.
Otherwise it is for variables.

n_par
*****
is the number of parameters.

vec_sets
========
#.  If i_par < n_par and set_id is in vec_sets.get_set(i_par),
    the parameter with index i_par depends on the parameter with index set_id.
#.  If set_id is in vec_sets.get_set(i_var + n_par) and set_id < n_par,
    the variable with index i_var depends on parameter with index set_id.
#.  If set_id is in vec_sets.get_set(i_var + n_par) and n_par <= set_id,
    the variable with index i_var depends on variable with index set_id - n_par.

On input, vec_sets is the dependencies for indices less than n_set.
If n_set < n_par,  n_result sets have been added, one for each
parameter result for this atomic function.
Otherwise, n_result sets have been added, one for each
variable result for this atomic function.

{xrst_end call_op_depend}
*/
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
    \
    atom_global_t&         atom_global   = atom_global_t::singleton(); \
    const base_atom_t&     base_atom = atom_global.get_base_atom(atom_id); \
    std::string            long_name = base_atom.long_name(call_info);
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
// ------------------------------------------------------------------------
// BEGIN_CALL_OP_DEPEND
void call_op_depend(
    size_t          op_index     ,
    const agraph_t& agraph       ,
    size_t          n_par        ,
    vec_sets_t&     vec_sets     )
{   // END_CALL_OP_DEPEND
    //
    assert( agraph.m_op_seq[op_index] == op_enum_t::call );
    //
    // parameter, variable
    ad_type_t parameter = ad_type_t::parameter;
    ad_type_t variable  = ad_type_t::variable;
    //
    // n_set
    size_t n_set = vec_sets.n_set();
    //
    // sub_sets
    thread_local vector<size_t>    sub_sets;
    //
    // arg_start. atom_id, call_info, n_domain, n_result
    size_t arg_start = agraph.m_arg_start[op_index];
    size_t atom_id   = agraph.m_arg_value[arg_start + 0];
    size_t call_info = agraph.m_arg_value[arg_start + 1];
    size_t n_domain  = agraph.m_arg_value[arg_start + 2];
    size_t n_result  = agraph.m_arg_value[arg_start + 4];
    //
    // long_name, base_atom
    atom_global_t&         atom_global   = atom_global_t::singleton();
    const base_atom_t&     base_atom  = atom_global.get_base_atom(atom_id);
    std::string            long_name  = base_atom.long_name(call_info);
    //
    // sparsity
    std::optional<sparsity_t> opt = base_atom.depend(call_info);
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".depend did not return a value\n";
        user_assert(false, msg);
    }
    sparsity_t sparsity = opt.value();
    sparsity.sort();
    //
    size_t sparsity_index = 0;
    size_t sparsity_row   = sparsity[0][0];
    for(size_t k = 0; k < n_result; ++k) {
        size_t arg_index    = arg_start + 5 + n_domain + k;
        size_t rng_index    = agraph.m_arg_value[arg_index];
        bool   more         = sparsity_index < sparsity.size();
        while(more && sparsity_row < rng_index) {
            ++sparsity_index;
            more  = sparsity_index < sparsity.size();
            if( more  ) {
                assert( sparsity_row <= sparsity[sparsity_index][0] );
                sparsity_row = sparsity[sparsity_index][0];
            }
        }
        sub_sets.resize(0);
        while(more && sparsity_row == rng_index) {
            size_t dom_index = sparsity[sparsity_index][1];
            if( n_domain <= dom_index ) {
                user_assert( dom_index < n_domain, long_name +
                    ".depend: sparsity column index is to large"
                );
            }
            arg_index           = arg_start + 5 + dom_index;
            ad_type_t arg_type  = agraph.m_arg_type[arg_index];
            size_t    arg_value = agraph.m_arg_value[arg_index];
            if( arg_type == parameter ) {
                sub_sets.push_back( arg_value );
            } else if( arg_type == variable ) {
                assert( n_par <= n_set );
                sub_sets.push_back( arg_value + n_par );
            }
            ++sparsity_index;
            more  = sparsity_index < sparsity.size();
            if( more  ) {
                assert( sparsity_row <= sparsity[sparsity_index][0] );
                sparsity_row = sparsity[sparsity_index][0];
            }
        }
#ifdef NDEBUG
        vec_sets.union_set(sub_sets);
#else
        size_t set_id = vec_sets.union_set(sub_sets);
        if( n_set < n_par ) {
            assert( set_id == op_index + k);
        } else {
            assert( set_id - n_par == op_index + k);
        }
    }
#endif
}
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
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    // long_name, base_atom
    UNPACK
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
    std::optional< vector<at::Tensor> > opt = base_atom.forward(
        call_info, rng_used, domain
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".forward did not return a value\n";
        user_assert(false, msg);
    }
    vector<at::Tensor> range = opt.value();
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
    // arg_start. atom_id, call_info, n_domain, n_range, n_result
    // long_name, base_atom
    UNPACK
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
    std::optional< vector<at::Tensor> > opt = base_atom.forward(
         call_info, rng_used, domain
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".forward did not return a value\n";
        user_assert(false, msg);
    }
    vector<at::Tensor> range = opt.value();
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
// TODO: figure out how to template this routine so do not need a different
// version for ad::Tensor and adten_t .
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
    // atom_callback, options
    UNPACK
    const atom_callback_t& atom_callback = atom_global.get_callback(atom_id);
    const options_t& options = atom_callback.get_options();
    //
    // forward_der
    const forward_der_t& forward_der = atom_callback.get_forward_der(call_info);
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
    std::optional< vector<at::Tensor> > opt = forward_der(
        options, call_info, rng_used, domain, dom_der
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".forward_der did not return a value\n";
        user_assert(false, msg);
    }
    vector<at::Tensor> rng_der = opt.value();
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
    // atom_callback, options
    UNPACK
    const atom_callback_t& atom_callback = atom_global.get_callback(atom_id);
    const options_t& options = atom_callback.get_options();
    //
    // forward_der
    const forward_der_t& forward_der =
        atom_callback.get_ad_forward_der(call_info);
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
    std::optional< vector<adten_t> > opt = forward_der(
        options, call_info, rng_used, domain, dom_der
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".ad_forward_der did not return a value\n";
        user_assert(false, msg);
    }
    vector<adten_t> rng_der = opt.value();
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
// TODO: figure out how to template this routine so do not need a different
// version for ad::Tensor and adten_t .
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
    // atom_callback, options
    UNPACK
    const atom_callback_t& atom_callback = atom_global.get_callback(atom_id);
    const options_t& options = atom_callback.get_options();
    //
    // reverse_der
    const reverse_der_t& reverse_der = atom_callback.get_reverse_der(call_info);
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
    std::optional< vector<at::Tensor> > opt = reverse_der(
        options, call_info, rng_used, domain, rng_der
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".reverse_der did not return a value\n";
        user_assert(false, msg);
    }
    vector<at::Tensor> dom_der = opt.value();
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
    // atom_callback, options
    UNPACK
    const atom_callback_t& atom_callback = atom_global.get_callback(atom_id);
    const options_t& options = atom_callback.get_options();
    //
    // reverse_der
    const reverse_der_t& reverse_der
        = atom_callback.get_ad_reverse_der(call_info);
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
    std::optional< vector<adten_t> > opt = reverse_der(
        options, call_info, rng_used, domain, rng_der
    );
    if( ! opt.has_value() ) {
        std::string msg = "atomic " + long_name;
        msg += ".ad_reverse_der did not return a value\n";
        user_assert(false, msg);
    }
    vector<adten_t> dom_der = opt.value();
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
