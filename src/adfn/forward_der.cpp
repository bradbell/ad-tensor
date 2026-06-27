// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/base_op.hpp>
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/dev/user_assert.hpp>
/*
{xrst_begin adfn_forward_der usr}
{xrst_spell
    rng
    dt
}

Compute A Domain Directional Derivative
#######################################

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_DER, END_FORWARD_DER
}

TensorType
**********
This is either at::Tensor or :ref:`adten-name` .

par_all
*******
is the value of all the parameters for this function.
This is usually calculated by :ref:`adfn_forward_par-name` .
In the special case where dom_par is empty, par_all is also empty
and need not be computed by adfn::forward_par .

var_all
*******
is the value of all the variables for this function.
This is usually calculated by :ref:`adfn_forward_var-name` .
Since derivatives are only computed with respect to domain variables,
it does not make sense for dom_var to be empty.

dom_der
*******
This is the domain direction that the derivative is computed with respect to.


options
*******
Trace the forward derivative calculations when options.get_trace() is true.

rng_der
*******
is the directional derivative of the range in the dom_der direction; i.e.

    rng_der = d/dt adfn(dom_par, dom_var + t * dom_der)


Example
*******
{xrst_literal ,
    examples/adfn/forward_der.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adfn_forward_der}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_FORWARD_DER
// rng_der = adfn.forward_der(par_all, var_all, dom_der, options)
template <class TensorType>
vector<TensorType> adfn_t::forward_der(
    const vector<TensorType>& par_all ,
    const vector<TensorType>& var_all ,
    const vector<TensorType>& dom_der ,
    const options_t&          options
) const
// END_FORWARD_DER
{
    // cout
    using std::cout;
    using std::string;
    using ad_tensor::dev::to_string;
    //
    // dom_der
# ifndef NDEBUg
    const vector< vector<int64_t> >&  shapes = m_var.m_dom_shapes;
    string msg = get_name() + ".forward_der: ";
    if( dom_der.size() != shapes.size() ) {
        msg += "dom_der.size() = " + std::to_string( dom_der.size() );
        msg += " and the dom_var size for this adfn is ";
        msg += std::to_string( shapes.size() );
        dev::user_assert( false , msg );
    }
    for(size_t i = 0; i < shapes.size(); ++i) {
        c10::IntArrayRef shape = shapes[i];
        if( ! dom_der[i].sizes().equals( shape ) ) {
            msg += "dom_der[" + std::to_string(i) + "] shape is ";
            msg += dev::to_string( dom_der[i].sizes() );
            msg += " and the dom_var shape for this index and adfn is ";
            msg += dev::to_string( shape );
            dev::user_assert( false , msg );
        }
    }
# endif
    //
    // trace
    bool trace = options.get_trace();
    if( trace ) {
        cout << "Begin tracing " + get_name() + ".forward_der\n";
    }
    //
    // n_op, n_all, empty
    size_t n_op      = m_var.m_op_seq.size();
    TensorType empty = TensorType( torch::empty( {0} ) );
    //
    // all_der
    vector<TensorType> all_der =  dom_der ;
    all_der.resize( n_op, empty );
    //
    // all_der
    for(size_t op_index = 0; op_index < n_op; ++op_index) {
        //
        // base_op
        dev::op_enum_t op_enum = m_var.m_op_seq[ op_index ];
        const dev::base_op_t<TensorType>& base_op =
            dev::op_enum2derive_op<TensorType>( op_enum );
        //
        // all_der
        base_op.forward_der(op_index, m_var, m_con, par_all, var_all, all_der);
        //
        if( trace) {
            string element = to_string( all_der[op_index] );
            cout << "all_der[" << op_index << "] = " << element;
            cout << ", " << to_string(op_enum)  << "(";
            size_t start = m_var.m_arg_start[op_index];
            size_t stop  = m_var.m_arg_start[op_index + 1];
            for(size_t i = start; i < stop; ++i) {
                cout << "[" << m_var.m_arg_value[i] << ",";
                cout << to_string( m_var.m_arg_type[i] ) << "]";
            }
            cout << ")\n";
        }
    }
    //
    // rng_der
    vector<TensorType> rng_der;
    TensorType zero = TensorType( torch::tensor( { 0.0 } ) );
    for(size_t i = 0; i < m_rng_index.size(); ++i) {
        size_t index = m_rng_index[i];
        ad_type_t ad_type  = m_rng_ad_type[i];
        switch(ad_type) {
            //
            case ad_type_t::constant: {
                c10::IntArrayRef shape = m_con[index].sizes();
                rng_der.push_back( TensorType(torch::zeros( shape ) ) );
            }
            break;
            //
            case ad_type_t::parameter: {
                c10::IntArrayRef shape = par_all[index].sizes();
                rng_der.push_back( TensorType( torch::zeros( shape ) ) );
            }
            break;
            //
            case ad_type_t::variable:
            rng_der.push_back( all_der[index] );
            break;
            //
            default:
            assert( false && "adfn.forward_der: "
                "a range type is not constant, parameter or variable"
            );
        }
    }
    if( trace ) {
        for(size_t i = 0; i < m_rng_index.size(); ++i) {
            string element = to_string( rng_der[i] );
            cout << "rng_der[" << i << "] = " << element << "\n";
        }
        cout << "End tracing " + get_name() + ".forward_der\n";
    }
    return rng_der;
}
template vector<at::Tensor> adfn_t::forward_der(
    const vector<at::Tensor>& par_all ,
    const vector<at::Tensor>& var_all ,
    const vector<at::Tensor>& dom_der ,
    const options_t&          options
) const;
template vector<adten_t> adfn_t::forward_der(
    const vector<adten_t>&    par_all ,
    const vector<adten_t>&    var_all ,
    const vector<adten_t>&    dom_der ,
    const options_t&          options
) const;

} // END_NAMESPACE_AD_TENSOR
